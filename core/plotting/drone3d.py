"""Depth-tested Drone6D plots using two independent OpenGL backends.

Camera adapted from LAVA-LAB/DynAbs, plotting/uav_plots.py:
https://github.com/LAVA-LAB/DynAbs/blob/main/plotting/uav_plots.py
Imports are lazy so other benchmarks do not need a graphics context.
"""

from itertools import islice
import asyncio
import logging
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# The old script uses (25, -35), but the supplied screenshot has a steeper
# view from the opposite side. These angles approximate its projected edges.
REFERENCE_VIEW = dict(elevation=60.0, azimuth=25.0)
WINDOW_SIZE = (1000, 900)
TRACE_COLOR = (0.36, 0.66, 1.0)


def plot_drone_3d_backends(args, stamp, idx_show, partition, model, traces, num_traces=100):
    """Run both renderers in fresh processes so native GL failures are isolated.

    Only NumPy arrays and plotting options cross the process boundary; the
    model/partition may otherwise contain large, non-picklable JAX objects.
    Call from a script protected by ``if __name__ == '__main__'``.
    """
    logger = logging.getLogger(__name__)
    options = SimpleNamespace(output_dir=str(getattr(args, 'output_dir', 'output')),
                              model=args.model, plot_title=getattr(args, 'plot_title', False))
    domain = SimpleNamespace(boundary_lb=np.asarray(partition.boundary_lb),
                             boundary_ub=np.asarray(partition.boundary_ub))
    regions = SimpleNamespace(**{name: [np.asarray(box) for box in getattr(model, name, [])]
                                 for name in ('goal', 'critical', 'charging_station')})
    selected = {i: {'x': np.asarray(trace['x'])}
                for i, trace in enumerate(islice(traces.values(), max(0, num_traces)))}
    results = {}
    context = multiprocessing.get_context('spawn')
    for renderer in (plot_traces_3d_pyvista, plot_traces_3d_visvis):
        name = renderer.__name__.removeprefix('plot_')
        process = context.Process(target=renderer,
                                  args=(options, stamp, list(idx_show), domain, regions, selected),
                                  kwargs={'num_traces': num_traces, 'filename': name})
        try:
            process.start()
            process.join()
            path = Path(options.output_dir) / f'{name}_{stamp}.png'
            if process.exitcode != 0 or not path.is_file():
                logger.error('%s failed (exit code %s); Matplotlib backup retained.',
                             renderer.__name__, process.exitcode)
                results[name] = None
            else:
                logger.info('Saved 3D plot to %s', path)
                results[name] = path
        except Exception:
            logger.exception('%s could not run; Matplotlib backup retained.', renderer.__name__)
            results[name] = None
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
    return results


def _camera_vectors(elevation=25.0, azimuth=-35.0):
    """Convert visvis's Rx(270+el) Rz(-az) view to VTK eye/up vectors."""
    el, az = np.deg2rad([elevation, azimuth])
    eye = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
    up = np.array([-np.sin(az) * np.sin(el), np.cos(az) * np.sin(el), np.cos(el)])
    return eye, up


def _camera_scale(low, high, aspect):
    """Fit all eight domain corners, leaving room for axis labels."""
    eye, up = _camera_vectors(**REFERENCE_VIEW)
    right = np.cross(up, eye)
    half_extent = (high - low) / 2
    return 1.18 * max(np.abs(up) @ half_extent, (np.abs(right) @ half_extent) / aspect)


def _ticks(low, high):
    raw = (high - low) / 10
    power = 10 ** np.floor(np.log10(raw))
    step = next(v * power for v in (1, 2, 5, 10) if v * power >= raw)
    return np.arange(np.ceil(low / step) * step, high + step * 1e-6, step)


def _pyvista_axes(plotter, pv, low, high):
    """Label three exterior edges with camera-facing text meshes.

    Avoid VTK cube-axes' duplicated labels and inconsistent text scaling
    during high-resolution screenshots. Text shares the scene's depth buffer.
    """
    eye, up = _camera_vectors(**REFERENCE_VIEW)
    right = np.cross(up, eye)
    rotation = np.column_stack((right, up, eye))
    size = np.max(high - low) / 34

    def text(label, position, height):
        mesh = pv.Text3D(label, depth=0, height=height, center=(0, 0, 0))
        mesh.points = mesh.points @ rotation.T + position
        plotter.add_mesh(mesh, color='black', lighting=False)

    for axis, offset in enumerate((-up, right, -right)):
        origin = low.copy()
        if axis == 1:
            origin[0] = high[0]
        for value in _ticks(low[axis], high[axis]):
            point = origin.copy()
            point[axis] = value
            plotter.add_mesh(pv.Line(point, point + offset * 0.15 * size), color='black')
            text(f'{value:g}', point + offset * 0.5 * size, 0.32 * size)
        midpoint = origin.copy()
        midpoint[axis] = (low[axis] + high[axis]) / 2
        text('XYZ'[axis], midpoint + offset * 1.3 * size, 0.5 * size)


def _scene(partition, model, idx_show, traces, num_traces):
    dims = np.asarray(idx_show, dtype=int)
    if dims.shape != (3,):
        raise ValueError('3D plots require exactly three state dimensions.')
    low = np.asarray(partition.boundary_lb, dtype=float)[dims]
    high = np.asarray(partition.boundary_ub, dtype=float)[dims]
    boxes = []
    for name, color in [('critical', (0.85, 0.02, 0.02)),
                        ('goal', (0.05, 0.85, 0.02)),
                        ('charging_station', (0.2, 0.8, 0.3))]:
        for box in getattr(model, name, []):
            boxes.append((np.asarray(box, dtype=float)[:, dims], color))
    paths = []
    for trace in islice(traces.values(), max(0, num_traces)):
        states = np.asarray(trace['x'], dtype=float)
        if len(states) == 0:
            continue
        points = states[:, dims]
        if not np.isfinite(points).all():
            raise ValueError('Trajectory coordinates must be finite.')
        # Repeated states are valid, but zero-length tube segments are not.
        keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
        paths.append(points[keep])
    return low, high, boxes, paths


def _tiled_box(low, high, color, tile_size=2.0):
    """Exterior quads with sharp normals and subtle, deterministic tile colors.

    Tiles are cosmetic (world units), not a projection of the 6D partition.
    Duplicated vertices keep adjacent faces flat-shaded in both backends.
    """
    vertices, colors = [], []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        us = np.linspace(low[u], high[u], max(1, int(np.ceil((high[u] - low[u]) / tile_size))) + 1)
        vs = np.linspace(low[v], high[v], max(1, int(np.ceil((high[v] - low[v]) / tile_size))) + 1)
        for side in (0, 1):
            for i in range(len(us) - 1):
                for j in range(len(vs) - 1):
                    quad = np.empty((4, 3))
                    quad[:, axis] = (low, high)[side][axis]
                    quad[:, u] = [us[i], us[i + 1], us[i + 1], us[i]]
                    quad[:, v] = [vs[j], vs[j], vs[j + 1], vs[j + 1]]
                    if side == 0:
                        quad = quad[::-1]
                    vertices.extend(quad)
                    colors.append(np.asarray(color) * (0.86 if (i + j) % 2 else 1.0))
    vertices = np.asarray(vertices, dtype=np.float32)
    return vertices, np.arange(len(vertices), dtype=np.uint32).reshape(-1, 4), np.asarray(colors)


def _output_path(args, stamp, filename):
    path = Path(getattr(args, 'output_dir', 'output')) / f'{filename}_{stamp}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_traces_3d_pyvista(args, stamp, idx_show, partition, model, traces,
                           num_traces=100, filename='traces_3d_pyvista'):
    """Save an off-screen PyVista rendering; return the PNG path."""
    import pyvista as pv

    low, high, boxes, paths = _scene(partition, model, idx_show, traces, num_traces)
    output = _output_path(args, stamp, filename)
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    try:
        plotter.set_background('white')
        plotter.enable_anti_aliasing('ssaa')
        for (bounds, color) in boxes:
            vertices, faces, colors = _tiled_box(*bounds, color)
            mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 4), faces])
            mesh.cell_data['colors'] = np.round(colors * 255).astype(np.uint8)
            plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=False,
                             ambient=0.3, diffuse=0.7, specular=0.0, show_scalar_bar=False)
        for points in paths:
            if len(points) > 1:
                tube = pv.lines_from_points(points).tube(radius=0.045, n_sides=8)
                plotter.add_mesh(tube, color=TRACE_COLOR, smooth_shading=True)
            plotter.add_points(points, color=TRACE_COLOR, point_size=5,
                               render_points_as_spheres=True)
            plotter.add_mesh(pv.Sphere(radius=0.12, center=points[0]), color='black')
        bounds = np.column_stack((low, high)).ravel()
        plotter.add_mesh(pv.Box(bounds=bounds).outline(), color='black', line_width=1)
        _pyvista_axes(plotter, pv, low, high)
        center = (low + high) / 2
        eye, up = _camera_vectors(REFERENCE_VIEW['elevation'], REFERENCE_VIEW['azimuth'])
        plotter.camera_position = (center + eye * np.linalg.norm(high - low) * 2, center, up)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = _camera_scale(low, high, WINDOW_SIZE[0] / WINDOW_SIZE[1])
        plotter.reset_camera_clipping_range()
        if getattr(args, 'plot_title', False):
            plotter.add_text(f'Simulation for {args.model}', font_size=14, color='black')
        plotter.screenshot(str(output), scale=3)
    finally:
        plotter.close()
    return output


def plot_traces_3d_visvis(args, stamp, idx_show, partition, model, traces,
                          num_traces=100, filename='traces_3d_visvis'):
    """Save a visvis rendering with the DynAbs camera, without app.Run().

    GLFW requires a desktop/OpenGL context, even when only exporting a PNG.
    """
    import visvis as vv
    import glfw
    from PIL import Image

    low, high, boxes, paths = _scene(partition, model, idx_show, traces, num_traces)
    output = _output_path(args, stamp, filename)
    app = vv.use('glfw')
    fig = vv.figure()
    try:
        fig.position.w, fig.position.h = WINDOW_SIZE
        fig.bgcolor = 'w'
        fig.relativeFontSize = 1.3
        ax = vv.gca()
        ax.cameraType = '3d'
        ax.daspectAuto = False
        ax.daspect = (1, 1, 1)
        for bounds, color in boxes:
            vertices, faces, colors = _tiled_box(*bounds, color)
            mesh = vv.mesh(vertices, faces, values=np.repeat(colors, 4, axis=0),
                           verticesPerFace=4, axes=ax, axesAdjust=False)
            mesh.faceShading = 'flat'
            mesh.ambient = 0.3
            mesh.diffuse = 0.7
            mesh.specular = 0.0
        for points in paths:
            vv.plot(*points.T, lw=2, lc=TRACE_COLOR, ms='.', mc=TRACE_COLOR,
                    mw=5, axes=ax, axesAdjust=False)
            vv.plot(*points[:1].T, lw=0, ms='o', mc='k', mw=8,
                    axes=ax, axesAdjust=False)
        ax.bgcolor = 'w'
        ax.axis.xLabel, ax.axis.yLabel, ax.axis.zLabel = 'X', 'Y', 'Z'
        ax.axis.xTicks, ax.axis.yTicks, ax.axis.zTicks = [
            list(_ticks(l, h)) for l, h in zip(low, high)]
        ax.axis.axisColor = 'k'
        ax.axis.showGrid = False
        ax.SetLimits(*list(zip(low, high)), margin=0)
        aspect = ax.position.width / ax.position.height
        scale = _camera_scale(low, high, aspect)
        ax.SetView(dict(REFERENCE_VIEW, zoom=1 / (2 * scale * min(1, aspect)),
                        fov=0, roll=0, loc=tuple((low + high) / 2)))
        if getattr(args, 'plot_title', False):
            vv.title(f'Simulation for {args.model}', axes=ax)
        fig.DrawNow()
        app.ProcessEvents()
        # GLFW schedules its paint callbacks on asyncio; ProcessEvents alone
        # only polls native events and otherwise yields an entirely black PNG.
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))
        pixels = vv.screenshot(None, sf=3, bg='w', ob=fig)
        if np.ptp(pixels) < 0.01:
            raise RuntimeError('visvis produced a blank frame; check the OpenGL context.')
        Image.fromarray(np.round(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(output)
    finally:
        # visvis 1.15 clears widget.figure before hiding the GLFW window.
        # On macOS the resulting focus event otherwise dereferences None.
        if fig._widget is not None:
            glfw.set_window_focus_callback(fig._widget._window, lambda *unused: None)
        vv.close(fig)
    return output

"""
Depth-tested Drone6D plots using PyVista/VTK.
"""

from itertools import islice
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import PchipInterpolator


# The old script uses (25, -35), but the supplied screenshot has a steeper
# view from the opposite side. These angles approximate its projected edges.
REFERENCE_VIEW = dict(elevation=60.0, azimuth=25.0)
WINDOW_SIZE = (1000, 600)
TRACE_COLOR = (0.36, 0.66, 1.0)


def _pyvista_worker(*args, **kwargs):
    """Render, report real failures, and exit before VTK module teardown.

    PyVista/VTK objects can participate in reference cycles. On Python 3.13,
    their destructors may run after PyVista's module globals have already been
    cleared, producing harmless ``Exception ignored in __del__`` tracebacks.
    This worker owns no state needed by its parent, so a direct process exit
    after the renderer's explicit cleanup avoids that broken teardown phase.
    """
    try:
        plot_traces_3d_pyvista(*args, **kwargs)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def plot_drone_3d_pyvista(args, stamp, idx_show, partition, model, traces, num_traces=100):
    """Run PyVista in a fresh process so native OpenGL failures are isolated.

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
    context = multiprocessing.get_context('spawn')
    name = 'traces_3d_pyvista'
    process = context.Process(target=_pyvista_worker,
                              args=(options, stamp, list(idx_show), domain, regions, selected),
                              kwargs={'num_traces': num_traces, 'filename': name})
    try:
        process.start()
        process.join()
        path = Path(options.output_dir) / f'{name}_{stamp}.png'
        if process.exitcode != 0 or not path.is_file():
            logger.error('plot_traces_3d_pyvista failed (exit code %s); Matplotlib backup retained.',
                         process.exitcode)
            return None
        logger.info('Saved 3D plot to %s', path)
        return path
    except Exception:
        logger.exception('plot_traces_3d_pyvista could not run; Matplotlib backup retained.')
        return None
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        process.close()


def _camera_vectors(elevation=25.0, azimuth=-35.0):
    """Convert elevation and azimuth to VTK eye/up vectors."""
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


def _pyvista_axes(plotter, pv, low, high):
    """Label the three axes with camera-facing text meshes.

    Text shares the scene's depth buffer and remains consistently scaled in
    high-resolution screenshots.
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
        midpoint = origin.copy()
        midpoint[axis] = (low[axis] + high[axis]) / 2
        text('XYZ'[axis], midpoint + offset * 1.3 * size, 1.25 * size)


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


def _smooth_path(points, samples_per_segment=4):
    """Interpolate a trajectory with a conservative, shape-preserving spline."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points

    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    samples = np.concatenate([
        np.linspace(distance[i], distance[i + 1], samples_per_segment, endpoint=False)
        for i in range(len(points) - 1)
    ] + [distance[-1:]])
    smooth = PchipInterpolator(distance, points, axis=0)(samples)
    smooth[::samples_per_segment] = points
    return smooth


def _tiled_box(low, high, color, tile_size=2.0):
    """Exterior quads with sharp normals and subtle, deterministic tile colors.

    Tiles are cosmetic (world units), not a projection of the 6D partition.
    Duplicated vertices keep adjacent faces flat-shaded.
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
    """Save an off-screen PyVista PNG rendering; return its path."""
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
                smooth_points = _smooth_path(points)
                tube = pv.lines_from_points(smooth_points).tube(radius=0.06, n_sides=8)
                plotter.add_mesh(tube, color=TRACE_COLOR, smooth_shading=True)
            plotter.add_points(points, color=TRACE_COLOR, point_size=5,
                               render_points_as_spheres=True)
            plotter.add_mesh(pv.Sphere(radius=0.12, center=points[0]), color='black')
        bounds = np.column_stack((low, high)).ravel()
        plotter.add_mesh(pv.Box(bounds=bounds).outline(), color='black', line_width=3)
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

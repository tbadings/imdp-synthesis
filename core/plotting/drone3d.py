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
# Match black at alpha=0.4 on the white background in traces.py.
TRACE_COLOR = (0.6, 0.6, 0.6)
MARKER_COLOR = (0,0,0)
# (model attribute, RGB, opacity); translucent regions let traces inside show.
REGION_STYLES = [('critical', (0.85, 0.02, 0.02), 1.0),
                 ('goal', (0.05, 0.85, 0.02), 1.0),
                 ('charging_station', (1.0, 0.78, 0.1), 0.35)]
DRONE_COLOR = (0.1, 0.25, 0.7)
# Battery gauge fill at 0%, 50%, and 100% charge.
BATTERY_COLORS = [(0.85, 0.1, 0.1), (1.0, 0.6, 0.0), (0.1, 0.7, 0.1)]
ANIMATION_FPS = 30
FRAMES_PER_STEP = 8

logger = logging.getLogger(__name__)


def _pyvista_worker(args, stamp, idx_show, partition, model, traces, num_traces, animate):
    """Render, report real failures, and exit before VTK module teardown.

    PyVista/VTK objects can participate in reference cycles. On Python 3.13,
    their destructors may run after PyVista's module globals have already been
    cleared, producing harmless ``Exception ignored in __del__`` tracebacks.
    This worker owns no state needed by its parent, so a direct process exit
    after the renderer's explicit cleanup avoids that broken teardown phase.
    A failed animation is reported but keeps exit code 0: the PNG is saved.
    """
    try:
        plot_traces_3d_pyvista(args, stamp, idx_show, partition, model, traces,
                               num_traces=num_traces)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    if animate:
        try:
            animate_trace_3d_pyvista(args, stamp, idx_show, partition, model, traces)
        except Exception:
            traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def plot_drone_3d_pyvista(args, stamp, idx_show, partition, model, traces, num_traces=100,
                          animate=True):
    """Run PyVista in a fresh process so native OpenGL failures are isolated.

    With ``animate``, also save a GIF of the first trace with its battery charge.

    Only NumPy arrays and plotting options cross the process boundary; the
    model/partition may otherwise contain large, non-picklable JAX objects.
    Call from a script protected by ``if __name__ == '__main__'``.
    """
    
    options = SimpleNamespace(output_dir=str(getattr(args, 'output_dir', 'output')),
                              model=args.model, plot_title=getattr(args, 'plot_title', False))
    domain = SimpleNamespace(boundary_lb=np.asarray(partition.boundary_lb),
                             boundary_ub=np.asarray(partition.boundary_ub))
    regions = SimpleNamespace(**{name: [np.asarray(box) for box in getattr(model, name, [])]
                                 for name in ('goal', 'critical', 'charging_station')})
    state_variables = list(getattr(model, 'state_variables', []))
    regions.battery_idx = state_variables.index('battery') if 'battery' in state_variables else None
    regions.max_charge = getattr(model, 'max_charge', None)
    selected = {i: {'x': np.asarray(trace['x'])}
                for i, trace in enumerate(islice(traces.values(), max(0, num_traces)))}
    context = multiprocessing.get_context('spawn')
    name = 'traces_3d_pyvista'
    process = context.Process(target=_pyvista_worker,
                              args=(options, stamp, list(idx_show), domain, regions, selected,
                                    num_traces, animate))
    try:
        process.start()
        process.join()
        path = Path(options.output_dir) / f'{name}_{stamp}.png'
        if process.exitcode != 0 or not path.is_file():
            logger.error('plot_traces_3d_pyvista failed (exit code %s); Matplotlib backup retained.',
                         process.exitcode)
            return None
        logger.info('Saved 3D plot to %s', path)
        if animate:
            gif = Path(options.output_dir) / f'traces_3d_pyvista_anim_{stamp}.gif'
            if gif.is_file():
                logger.info('Saved 3D animation to %s', gif)
            else:
                logger.error('animate_trace_3d_pyvista failed; see the traceback above.')
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
    for name, color, opacity in REGION_STYLES:
        for box in getattr(model, name, []):
            boxes.append((np.asarray(box, dtype=float)[:, dims], color, opacity))
    paths = []
    for i, trace in enumerate(islice(traces.values(), max(0, num_traces))):
        states = np.asarray(trace['x'], dtype=float)
        if len(states) == 0:
            continue

        print(f'- Trace {i}, number of steps: {len(states)} start: {states[0]}, end: {states[-1]}', flush=True)

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


def _output_path(args, stamp, filename, suffix='.png'):
    path = Path(getattr(args, 'output_dir', 'output')) / f'{filename}_{stamp}{suffix}'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _draw_scene(plotter, pv, args, low, high, boxes):
    """Regions, domain outline, axis labels, camera, and optional title."""
    plotter.set_background('white')
    plotter.enable_anti_aliasing('ssaa')
    plotter.enable_depth_peeling()
    for (bounds, color, opacity) in boxes:
        if opacity < 1:
            # Plain faces (tiles would show through each other) and darker edges.
            box = pv.Box(bounds=bounds.T.ravel())
            plotter.add_mesh(box, color=color, opacity=opacity, smooth_shading=False,
                             ambient=0.3, diffuse=0.7, specular=0.0)
            plotter.add_mesh(box.outline(), color=np.asarray(color) * 0.75, line_width=3)
            continue
        vertices, faces, colors = _tiled_box(*bounds, color)
        mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 4), faces])
        mesh.cell_data['colors'] = np.round(colors * 255).astype(np.uint8)
        plotter.add_mesh(mesh, scalars='colors', rgb=True, smooth_shading=False,
                         ambient=0.3, diffuse=0.7, specular=0.0, show_scalar_bar=False)
    bounds = np.column_stack((low, high)).ravel()
    plotter.add_mesh(pv.Box(bounds=bounds).outline(), color='black', line_width=3)
    _pyvista_axes(plotter, pv, low, high)
    center = (low + high) / 2
    eye, up = _camera_vectors(REFERENCE_VIEW['elevation'], REFERENCE_VIEW['azimuth'])
    plotter.camera_position = (center + eye * np.linalg.norm(high - low) * 2, center, up)
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = _camera_scale(low, high, WINDOW_SIZE[0] / WINDOW_SIZE[1])
    if getattr(args, 'plot_title', False):
        plotter.add_text(f'Simulation for {args.model}', font_size=14, color='black')


def _markers(pv, points):
    """Mark the original samples, not the interpolated path points.

    World-space spheres stay wider than the tube at every resolution.
    """
    return pv.PolyData(points).glyph(
        geom=pv.Sphere(radius=0.25, theta_resolution=12, phi_resolution=12),
        orient=False, scale=False,
    )


def plot_traces_3d_pyvista(args, stamp, idx_show, partition, model, traces,
                           num_traces=100, filename='traces_3d_pyvista'):
    """Save an off-screen PyVista PNG rendering; return its path."""
    import pyvista as pv

    low, high, boxes, paths = _scene(partition, model, idx_show, traces, num_traces)
    output = _output_path(args, stamp, filename)
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    try:
        _draw_scene(plotter, pv, args, low, high, boxes)
        for points in paths:
            if len(points) > 1:
                smooth_points = _smooth_path(points)
                tube = pv.lines_from_points(smooth_points).tube(radius=0.15, n_sides=8)
                plotter.add_mesh(tube, color=TRACE_COLOR, smooth_shading=True)
            plotter.add_mesh(_markers(pv, points), color=MARKER_COLOR, smooth_shading=True)
            plotter.add_mesh(pv.Sphere(radius=0.22, center=points[0]), color='black')
        plotter.reset_camera_clipping_range()
        plotter.screenshot(str(output), scale=3)
    finally:
        plotter.close()
    return output


def _drone_mesh(pv, size):
    """Level quadrotor centred at the origin, about 2.3 * size across."""
    arm = 0.75 * size
    parts = [pv.Box(bounds=0.25 * size * np.array([-1, 1, -1, 1, -0.5, 0.5]))]
    for angle in (45, 135):
        parts.append(pv.Cube(x_length=2 * arm, y_length=0.12 * size, z_length=0.08 * size)
                     .rotate_z(angle, inplace=False))
    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        center = (dx * arm / np.sqrt(2), dy * arm / np.sqrt(2), 0.06 * size)
        parts.append(pv.Cylinder(center=center, direction=(0, 0, 1), radius=0.42 * size,
                                 height=0.04 * size, resolution=24))
    return pv.merge(parts)


def _battery_color(charge):
    """Red when empty, orange at half, green when full."""
    return tuple(np.interp(charge, [0, 0.5, 1], channel)
                 for channel in zip(*BATTERY_COLORS))


def _battery_gauge(plotter, pv, origin, basis, size, charge):
    """Camera-facing battery icon with the charge in percent, left edge at ``origin``.

    ``basis`` holds the screen right/up and view directions as columns. The
    layers are separated along the view direction, which does not move them on
    screen under parallel projection but avoids z-fighting.
    """
    w, h, t = 3.4 * size, 1.4 * size, 0.2 * size
    layer = 0.05 * size

    def quad(name, u0, v0, u1, v1, color, depth):
        local = np.array([[u0, v0, depth], [u1, v0, depth], [u1, v1, depth], [u0, v1, depth]])
        plotter.add_mesh(pv.PolyData(local @ basis.T + origin, faces=[4, 0, 1, 2, 3]),
                         color=color, lighting=False, name=name)

    quad('battery_body', 0, -h / 2, w, h / 2, 'black', 0)
    quad('battery_tip', w, -h / 4, w + 1.5 * t, h / 4, 'black', 0)
    quad('battery_inner', t, -h / 2 + t, w - t, h / 2 - t, 'white', layer)
    fill = 2 * t + (w - 4 * t) * charge
    quad('battery_fill', 2 * t, -h / 2 + 2 * t, fill, h / 2 - 2 * t, _battery_color(charge), 2 * layer)
    text = pv.Text3D(f'{100 * charge:.0f}%', depth=0, height=0.9 * h, center=(0, 0, 0))
    text.points += [w + 3 * t + (text.bounds[1] - text.bounds[0]) / 2, 0, 0]
    text.points = text.points @ basis.T + origin
    plotter.add_mesh(text, color='black', lighting=False, name='battery_text')


def _gif_palette(frame):
    """One palette for every frame: no colour shimmer, and frames compress as deltas.

    The first frame lacks the trail and most gauge colours, so append ramps of them.
    """
    from PIL import Image
    ramps = [np.outer(np.linspace(0, 1, 64), TRACE_COLOR) / max(TRACE_COLOR),
             np.outer(np.linspace(0, 1, 64), DRONE_COLOR) / max(DRONE_COLOR),
             np.array([_battery_color(c) for c in np.linspace(0, 1, 64)])]
    swatch = np.repeat(np.concatenate(ramps)[None], frame.height // 10, axis=0)
    swatch = Image.fromarray(np.round(swatch * 255).astype(np.uint8)).resize(
        (frame.width, frame.height // 10), Image.Resampling.NEAREST)
    combined = Image.new('RGB', (frame.width, frame.height + swatch.height))
    combined.paste(frame)
    combined.paste(swatch, (0, frame.height))
    palette = combined.quantize(256, method=Image.Quantize.MEDIANCUT)
    # Median cut averages anti-aliased edges into white and black. Pillow maps
    # pixels to palette entries at 5 bits per channel, so also snap the near
    # neighbours, or pure white may be drawn as one of them.
    colors = np.array(palette.getpalette()).reshape(-1, 3)
    colors[(colors >= 248).all(axis=1)] = 255
    colors[(colors <= 7).all(axis=1)] = 0
    palette.putpalette(colors.astype(np.uint8).ravel().tolist())
    return palette


def animate_trace_3d_pyvista(args, stamp, idx_show, partition, model, traces, trace_index=0,
                             filename='traces_3d_pyvista_anim', fps=ANIMATION_FPS,
                             frames_per_step=FRAMES_PER_STEP, hold_seconds=1.5):
    """Save a GIF of one drone flying its trace, with its battery charge alongside; return its path.

    The charge gauge is shown when ``model.battery_idx`` is set; ``model.max_charge``
    maps it to percent.
    """
    import pyvista as pv
    from PIL import Image

    dims = np.asarray(idx_show, dtype=int)
    low, high, boxes, _ = _scene(partition, model, idx_show, {}, 0)
    states = np.asarray(list(traces.values())[trace_index]['x'], dtype=float)
    steps = np.arange(len(states))
    times = np.linspace(0, steps[-1], steps[-1] * frames_per_step + 1)
    # Interpolate over time, not arc length, so hovering (e.g. while charging) takes time.
    positions = (PchipInterpolator(steps, states[:, dims], axis=0)(times)
                 if len(states) > 1 else states[:, dims])
    battery_idx = getattr(model, 'battery_idx', None)
    charge = (None if battery_idx is None else
              np.clip(np.interp(times, steps, states[:, battery_idx]) / model.max_charge, 0, 1))

    eye, up = _camera_vectors(**REFERENCE_VIEW)
    basis = np.column_stack((np.cross(up, eye), up, eye))
    size = np.max(high - low) / 34
    # Toward the camera by the domain diagonal: in front of every obstacle,
    # yet at the same screen position under parallel projection.
    gauge_offset = basis @ [1.6 * size, 1.2 * size, np.linalg.norm(high - low)]
    output = _output_path(args, stamp, filename, suffix='.gif')
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    frames, palette = [], None
    try:
        _draw_scene(plotter, pv, args, low, high, boxes)
        drone = plotter.add_mesh(_drone_mesh(pv, size), color=DRONE_COLOR, smooth_shading=False,
                                 ambient=0.3, diffuse=0.7, specular=0.0)
        for k, position in enumerate(positions):
            trail = positions[:k + 1]
            # Repeated states are valid, but zero-length tube segments are not.
            trail = trail[np.r_[True, np.any(np.diff(trail, axis=0) != 0, axis=1)]]
            if len(trail) > 1:
                plotter.add_mesh(pv.lines_from_points(trail).tube(radius=0.15, n_sides=8),
                                 color=TRACE_COLOR, smooth_shading=True, name='trail')
            plotter.add_mesh(_markers(pv, states[:k // frames_per_step + 1, dims]),
                             color=MARKER_COLOR, smooth_shading=True, name='markers')
            drone.position = position
            if charge is not None:
                _battery_gauge(plotter, pv, position + gauge_offset, basis, size, charge[k])
            plotter.reset_camera_clipping_range()
            frame = Image.fromarray(plotter.screenshot(return_img=True))
            if palette is None:
                palette = _gif_palette(frame)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
    finally:
        plotter.close()
    durations = [round(1000 / fps)] * (len(frames) - 1) + [round(1000 * hold_seconds)]
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return output

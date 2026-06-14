# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Render a Kimodo motion NPZ to an MP4.

The NPZ layout matches what create_data.py / create_benchmark.py produce:
posed_joints (T, 77, 3), global_rot_mats (T, 77, 3, 3), root_positions (T, 3), ...

Three starting camera presets are available: 'general' (3/4 view), 'top' (overhead),
and 'front' (chest-height looking at the canonicalized facing direction). The camera
translates with the root each frame so the character stays centered.

Example:
    python -m kimodo.scripts.visualize motion.npz --view general,top,front --fps 30
"""

import argparse
import os
from pathlib import Path

# Must set the GL platform before importing pyrender.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# pyrender 0.1.45 still uses np.infty, which was removed in NumPy 2.0. Shim it back
# (no-op if NumPy already has it). Required before pyrender is imported.
import numpy as _np  # noqa: E402

if not hasattr(_np, "infty"):
    _np.infty = _np.inf


def _patch_pyrender_egl_device_pick() -> None:
    """Make pyrender's EGL device lookup pick the first device that actually
    initializes. Default pyrender takes ``query_devices()[0]``, which on cluster
    nodes is often a mesa device that needs ``/dev/dri/...`` access and fails;
    we iterate and skip those so we land on the NVIDIA-direct (or other working)
    device. If no devices initialize, fall through to the original default
    (which uses ``EGL_DEFAULT_DISPLAY`` via libglvnd).
    """
    if os.environ.get("PYOPENGL_PLATFORM") != "egl":
        return
    try:
        import ctypes as _ctypes

        from pyrender.platforms import egl as _pyr_egl
        from OpenGL import EGL as _egl
    except Exception:
        return

    def _pick_working_device():
        try:
            devices = _pyr_egl.query_devices()
        except Exception:
            return _pyr_egl.EGLDevice(None)
        if not devices:
            return _pyr_egl.EGLDevice(None)
        for dev in devices:
            try:
                display = dev.get_display()
                if not display:
                    continue
                major = _ctypes.c_long()
                minor = _ctypes.c_long()
                if _egl.eglInitialize(display, major, minor):
                    _egl.eglTerminate(display)
                    return dev
            except Exception:
                continue
        # None of the device-specific displays initialized cleanly; fall back to
        # the libglvnd default display, which may still work on some setups.
        return _pyr_egl.EGLDevice(None)

    _pyr_egl.get_default_device = _pick_working_device

    _orig_get_by_index = _pyr_egl.get_device_by_index

    def _patched_get_by_index(device_id):
        # OffscreenRenderer hard-codes id=0; treat that as "pick the working one".
        if device_id == 0:
            return _pick_working_device()
        return _orig_get_by_index(device_id)

    _pyr_egl.get_device_by_index = _patched_get_by_index


_patch_pyrender_egl_device_pick()

import imageio.v3 as iio
import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm

from kimodo.skeleton import SOMASkeleton77
from kimodo.viz.soma_skin import SOMASkin


# Per-view: camera offset from the (tracked) root in world units, the height of the
# look-at target relative to the root, and the "up" hint passed to look_at.
VIEW_PRESETS: dict[str, dict] = {
    "general": dict(offset=(3.0, 2.5, 3.0), look_at_height=1.0, up=(0.0, 1.0, 0.0)),
    "top": dict(offset=(0.0, 6.5, 0.0), look_at_height=0.0, up=(0.0, 0.0, 1.0)),
    "front": dict(offset=(0.0, 1.8, 4.5), look_at_height=1.2, up=(0.0, 1.0, 0.0)),
}

# Vertical FOV shared by the renderer and the fixed-camera framing solver.
_YFOV = np.pi / 4.5


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed look-at matrix in pyrender (OpenGL) convention: -Z forward, +Y up."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # up is parallel to fwd; nudge it.
        up = up + np.array([1e-3, 0.0, 1e-3])
        right = np.cross(fwd, up)
        right_norm = np.linalg.norm(right)
    right /= right_norm
    new_up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = new_up
    pose[:3, 2] = -fwd
    pose[:3, 3] = eye
    return pose


# -----------------------------------------------------------------------------
# Constraint overlay + fixed (non-tracking) camera
# -----------------------------------------------------------------------------
# Per-end-effector marker colors (RGBA). Anything not listed uses EE_DEFAULT.
_EE_COLORS = {
    "LeftHand": (235, 70, 70, 255),
    "RightHand": (70, 120, 235, 255),
    "LeftFoot": (70, 205, 95, 255),
    "RightFoot": (225, 90, 205, 255),
}
_EE_DEFAULT = (235, 70, 70, 255)
_ROOT_COLOR = (245, 165, 45, 255)      # root 2D path (orange)
_FULLBODY_COLOR = (70, 200, 200, 255)  # full-body keyframe ghost joints (cyan)


def _sphere(center, radius: float, color_rgba) -> trimesh.Trimesh:
    m = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    m.apply_translation(np.asarray(center, dtype=np.float64))
    m.visual.vertex_colors = np.tile(np.asarray(color_rgba, dtype=np.uint8), (len(m.vertices), 1))
    return m


def _cylinder_between(p0, p1, radius: float, color_rgba):
    """A capsule-free cylinder spanning p0->p1 (used for root-path segments)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    seg = p1 - p0
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        return None
    m = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
    # cylinder is +Z aligned & origin-centered; rotate +Z onto the segment direction.
    z = np.array([0.0, 0.0, 1.0])
    d = seg / length
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, d))
    transform = np.eye(4)
    if s < 1e-8:
        if c < 0:  # antiparallel: flip about X
            transform[:3, :3] = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
    else:
        transform[:3, :3] = trimesh.transformations.rotation_matrix(np.arctan2(s, c), axis / s)[:3, :3]
    transform[:3, 3] = (p0 + p1) / 2.0
    m.apply_transform(transform)
    m.visual.vertex_colors = np.tile(np.asarray(color_rgba, dtype=np.uint8), (len(m.vertices), 1))
    return m


def build_constraint_meshes(constraints_lst, skeleton, start: int, end: int):
    """Turn loaded constraint sets into a single colored trimesh of world-fixed markers.

    Returns ``(combined_trimesh_or_None, marker_points (P,3))``. Markers:
      * root2d        -> orange spheres on the floor (y~0) + connecting path tube
      * end-effector  -> per-joint colored spheres at the target EE positions
      * fullbody      -> small cyan spheres at every target joint (ghost pose)
    Only keyframes whose frame index falls in [start, end) are drawn.
    """
    meshes: list[trimesh.Trimesh] = []
    pts: list[list[float]] = []
    for c in constraints_lst:
        name = getattr(c, "name", "")
        fi = c.frame_indices.detach().cpu().numpy().astype(int)
        keep = (fi >= start) & (fi < end)
        if name == "root2d":
            xz = c.smooth_root_2d.detach().cpu().numpy()
            order = np.argsort(fi)
            path = []
            for k in order:
                if not keep[k]:
                    continue
                p = [float(xz[k, 0]), 0.04, float(xz[k, 1])]
                path.append(p)
                pts.append(p)
                meshes.append(_sphere(p, 0.06, _ROOT_COLOR))
            for a, b in zip(path[:-1], path[1:]):
                seg = _cylinder_between(a, b, 0.02, _ROOT_COLOR)
                if seg is not None:
                    meshes.append(seg)
        elif name == "fullbody":
            gp = c.global_joints_positions.detach().cpu().numpy()  # (N, J, 3)
            for k in range(len(fi)):
                if not keep[k]:
                    continue
                for j in range(gp.shape[1]):
                    p = gp[k, j].tolist()
                    pts.append(p)
                    meshes.append(_sphere(p, 0.035, _FULLBODY_COLOR))
        else:  # end-effector family (end-effector / left-hand / right-foot / ...)
            gp = c.global_joints_positions.detach().cpu().numpy()  # (N, J, 3)
            # Mark only the NAMED end-effectors (e.g. LeftHand/RightHand), not the
            # full expanded sub-chain that pos_indices covers (which is ~50 joints).
            jnames = getattr(c, "joint_names", None)
            if jnames:
                ee_targets = [
                    (skeleton.bone_index[n], _EE_COLORS.get(n, _EE_DEFAULT))
                    for n in jnames if n in skeleton.bone_index
                ]
            else:
                ee_targets = [(int(j), _EE_DEFAULT) for j in c.pos_indices.detach().cpu().numpy()]
            for k in range(len(fi)):
                if not keep[k]:
                    continue
                for j, col in ee_targets:
                    p = gp[k, int(j)].tolist()
                    pts.append(p)
                    meshes.append(_sphere(p, 0.07, col))
    if not meshes:
        return None, np.zeros((0, 3), dtype=np.float64)
    combined = trimesh.util.concatenate(meshes)
    return combined, np.asarray(pts, dtype=np.float64)


def compute_fixed_camera_pose(preset: dict, frame_points: np.ndarray, marker_points: np.ndarray,
                              yfov: float, aspect: float, margin: float = 1.25) -> np.ndarray:
    """One static camera pose that frames the whole motion (+ constraint markers).

    Uses the preset's offset only as a *direction*; distance is solved so the
    bounding sphere of all points fits the (smaller of vertical/horizontal) FOV.
    The camera then does NOT track the root, so the floor stays put and the
    character translates across it.
    """
    pts = frame_points.reshape(-1, 3)
    if marker_points.size:
        pts = np.concatenate([pts, marker_points], axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2.0
    radius = max(float(np.linalg.norm(pts - center, axis=1).max()), 1.0)
    hfov = 2.0 * np.arctan(np.tan(yfov / 2.0) * aspect)
    half = min(yfov, hfov) / 2.0
    dist = radius / max(np.sin(half), 1e-3) * margin
    direction = np.asarray(preset["offset"], dtype=np.float64)
    direction /= max(np.linalg.norm(direction), 1e-9)
    eye = center + direction * dist
    return look_at(eye, center, np.asarray(preset["up"], dtype=np.float64))


def compute_skinned_vertices(
    skin: SOMASkin,
    joints_rot: torch.Tensor,
    joints_pos: torch.Tensor,
    rot_is_global: bool,
    chunk: int = 32,
) -> np.ndarray:
    """Apply LBS in chunks to keep peak VRAM bounded."""
    T = joints_pos.shape[0]
    chunks = []
    with torch.no_grad():
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            verts = skin.skin(
                joints_rot[start:end],
                joints_pos[start:end],
                rot_is_global=rot_is_global,
            )
            chunks.append(verts.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def _make_checker_floor(half_extent: float = 20.0, tiles: int = 20) -> trimesh.Trimesh:
    """Per-tile colored checker floor at y=0, so we can see character motion across it."""
    cell = (2 * half_extent) / tiles
    verts = []
    faces = []
    colors = []
    light = np.array([200, 205, 215, 255], dtype=np.uint8)
    dark = np.array([170, 175, 190, 255], dtype=np.uint8)
    for i in range(tiles):
        for j in range(tiles):
            x0 = -half_extent + i * cell
            z0 = -half_extent + j * cell
            base = len(verts)
            verts.extend([
                [x0, 0.0, z0],
                [x0 + cell, 0.0, z0],
                [x0 + cell, 0.0, z0 + cell],
                [x0, 0.0, z0 + cell],
            ])
            # CCW from above so face normals point +Y (up toward the lights).
            faces.append([base, base + 2, base + 1])
            faces.append([base, base + 3, base + 2])
            c = light if (i + j) % 2 == 0 else dark
            colors.extend([c, c, c, c])
    tm = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        vertex_colors=np.array(colors, dtype=np.uint8),
        process=False,
    )
    return tm


def build_static_scene(
    initial_vertices: np.ndarray,
    faces: np.ndarray,
    body_color=(152, 189, 255, 255),
    constraint_mesh: "trimesh.Trimesh | None" = None,
) -> tuple[pyrender.Scene, pyrender.Node]:
    scene = pyrender.Scene(
        bg_color=(0.92, 0.94, 0.97, 1.0),
        ambient_light=(0.30, 0.30, 0.30),
    )

    # Checker floor at y=0 so motion is easy to read.
    ground_tm = _make_checker_floor(half_extent=20.0, tiles=40)
    scene.add(pyrender.Mesh.from_trimesh(ground_tm, smooth=False))

    # World-fixed constraint markers (added once; they don't move).
    if constraint_mesh is not None:
        scene.add(pyrender.Mesh.from_trimesh(constraint_mesh, smooth=True))

    # Character mesh node (we'll swap it out each frame).
    char_tm = trimesh.Trimesh(vertices=initial_vertices, faces=faces, process=False)
    char_tm.visual.vertex_colors = np.tile(np.array(body_color, dtype=np.uint8), (len(char_tm.vertices), 1))
    char_node = scene.add(pyrender.Mesh.from_trimesh(char_tm, smooth=True))

    # Key light (high + slightly behind-right) -- this is the shadow caster.
    scene.add(pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=3.5),
              pose=look_at(np.array([4, 9, 4]), np.array([0, 1, 0]), np.array([0, 1, 0])))
    # Fill from opposite side.
    scene.add(pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=1.4),
              pose=look_at(np.array([-5, 4, -3]), np.array([0, 1, 0]), np.array([0, 1, 0])))
    # Rim from behind.
    scene.add(pyrender.DirectionalLight(color=(1.0, 1.0, 1.0), intensity=0.8),
              pose=look_at(np.array([0, 4, -6]), np.array([0, 1, 0]), np.array([0, 1, 0])))

    return scene, char_node


def render_motion(
    verts_all: np.ndarray,
    faces: np.ndarray,
    root_xz_per_frame: np.ndarray,
    preset: dict,
    width: int,
    height: int,
    camera_mode: str = "follow",
    fixed_cam_pose: "np.ndarray | None" = None,
    constraint_mesh: "trimesh.Trimesh | None" = None,
) -> list[np.ndarray]:
    """Render frames.

    camera_mode:
      * "follow" (default): camera tracks the root XZ each frame -> character
        stays centered, floor appears to slide (original behavior).
      * "fixed": one static camera (``fixed_cam_pose``) frames the whole clip ->
        the floor stays put and the character moves across it. Required to read
        world-fixed constraint markers correctly.
    """
    scene, char_node = build_static_scene(verts_all[0], faces, constraint_mesh=constraint_mesh)

    yfov = _YFOV
    cam = pyrender.PerspectiveCamera(yfov=yfov, znear=0.05, zfar=200.0)
    cam_node = scene.add(cam, pose=np.eye(4))
    if camera_mode == "fixed" and fixed_cam_pose is not None:
        scene.set_pose(cam_node, fixed_cam_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    render_flags = pyrender.RenderFlags.SHADOWS_DIRECTIONAL

    offset = np.asarray(preset["offset"], dtype=np.float64)
    look_h = float(preset["look_at_height"])
    up_hint = np.asarray(preset["up"], dtype=np.float64)

    T = verts_all.shape[0]
    frames: list[np.ndarray] = []
    body_color = np.tile(np.array([152, 189, 255, 255], dtype=np.uint8), (verts_all.shape[1], 1))
    try:
        for t in tqdm(range(T), desc="rendering"):
            char_tm = trimesh.Trimesh(vertices=verts_all[t], faces=faces, process=False)
            char_tm.visual.vertex_colors = body_color
            scene.remove_node(char_node)
            char_node = scene.add(pyrender.Mesh.from_trimesh(char_tm, smooth=True))

            if camera_mode != "fixed":
                rx, rz = float(root_xz_per_frame[t, 0]), float(root_xz_per_frame[t, 1])
                target = np.array([rx, look_h, rz])
                eye = np.array([rx + offset[0], offset[1], rz + offset[2]])
                scene.set_pose(cam_node, look_at(eye, target, up_hint))

            color, _ = renderer.render(scene, flags=render_flags)
            frames.append(color)
    finally:
        renderer.delete()

    return frames


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="Path to a Kimodo motion NPZ (gt_motion.npz, motion.npz, or output of create_data.py).")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output MP4 path. With multiple --view modes, this becomes the prefix and '_<view>.mp4' is appended. Default: <npz>_<view>.mp4 next to the NPZ.",
    )
    parser.add_argument(
        "--view",
        default="general",
        help="Comma-separated list of starting views from {general, top, front}. One MP4 is written per view.",
    )
    parser.add_argument("--width", type=int, default=960, help="Output width in pixels (rounded up to a multiple of 16).")
    parser.add_argument("--height", type=int, default=720, help="Output height in pixels (rounded up to a multiple of 16).")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (matches how the NPZ was sampled; default 30).")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for skinning ('cuda' or 'cpu').",
    )
    parser.add_argument("--start", type=int, default=0, help="First frame to render (default 0).")
    parser.add_argument("--end", type=int, default=None, help="Last frame to render (exclusive). Default: end of motion.")
    parser.add_argument(
        "--constraints",
        type=Path,
        default=None,
        help="Path to a constraints.json to overlay as world-fixed markers (root path / "
        "end-effector / full-body keyframes). Defaults to 'constraints.json' next to the NPZ "
        "if present. Implies --camera fixed unless overridden.",
    )
    parser.add_argument(
        "--camera",
        choices=("follow", "fixed"),
        default=None,
        help="'follow' (default w/o constraints): camera tracks the root, character stays "
        "centered. 'fixed' (default w/ constraints): static camera frames the whole clip so the "
        "floor stays put and the character moves across it.",
    )
    args = parser.parse_args()

    views = [v.strip() for v in args.view.split(",") if v.strip()]
    for v in views:
        if v not in VIEW_PRESETS:
            raise SystemExit(f"Unknown view '{v}'. Choose from: {sorted(VIEW_PRESETS)}.")

    if not args.npz.is_file():
        raise SystemExit(f"NPZ not found: {args.npz}")

    data = dict(np.load(args.npz, allow_pickle=True))
    required = {"posed_joints"}
    missing = required - set(data.keys())
    if missing:
        raise SystemExit(f"NPZ missing keys: {sorted(missing)}. Found: {sorted(data.keys())}")

    posed_joints_np = data["posed_joints"]  # (T, J, 3)
    T_all, J, _ = posed_joints_np.shape
    start = max(0, args.start)
    end = T_all if args.end is None else min(T_all, args.end)
    if end <= start:
        raise SystemExit(f"Empty frame range [{start}, {end}).")

    joints_pos = torch.from_numpy(posed_joints_np[start:end]).float().to(args.device)
    if "global_rot_mats" in data:
        joints_rot = torch.from_numpy(data["global_rot_mats"][start:end]).float().to(args.device)
        rot_is_global = True
    elif "local_rot_mats" in data:
        joints_rot = torch.from_numpy(data["local_rot_mats"][start:end]).float().to(args.device)
        rot_is_global = False
    else:
        raise SystemExit("NPZ must contain either 'global_rot_mats' or 'local_rot_mats'.")

    if "root_positions" in data:
        root_xz = data["root_positions"][start:end][:, [0, 2]]
    else:
        root_xz = posed_joints_np[start:end][:, 0, [0, 2]]

    skeleton = SOMASkeleton77().to(args.device)
    skin = SOMASkin(skeleton)
    print(f"Skinning {joints_pos.shape[0]} frames on {args.device} ...")
    verts_all = compute_skinned_vertices(skin, joints_rot, joints_pos, rot_is_global=rot_is_global)
    faces = skin.faces.cpu().numpy()
    print(f"  vertices: {verts_all.shape}, faces: {faces.shape}")

    # ---- optional constraint overlay + fixed (non-tracking) camera ----
    constraints_path = args.constraints
    if constraints_path is None:
        cand = args.npz.parent / "constraints.json"
        if cand.is_file():
            constraints_path = cand
    constraint_mesh = None
    marker_points = np.zeros((0, 3), dtype=np.float64)
    if constraints_path is not None and Path(constraints_path).is_file():
        from kimodo.constraints import load_constraints_lst

        cons_skel = SOMASkeleton77()  # CPU skeleton; from_dict runs FK -> world targets
        cons = load_constraints_lst(str(constraints_path), cons_skel, device="cpu")
        constraint_mesh, marker_points = build_constraint_meshes(cons, cons_skel, start, end)
        types = ", ".join(getattr(c, "name", "?") for c in cons)
        print(f"  constraints: {len(cons)} set(s) [{types}] from {Path(constraints_path).name}; "
              f"{marker_points.shape[0]} marker(s)")
    elif args.constraints is not None:
        print(f"  WARNING: --constraints file not found: {args.constraints}")

    # Camera mode: explicit flag wins; otherwise 'fixed' when constraints are present.
    camera_mode = args.camera or ("fixed" if constraint_mesh is not None else "follow")
    frame_points = posed_joints_np[start:end].reshape(-1, 3).astype(np.float64)

    width = _round_up(args.width, 16)
    height = _round_up(args.height, 16)

    # Decide how to interpret --output:
    #   - existing directory or trailing slash      -> save as <out_dir>/<npz_stem>_<view>.mp4
    #   - explicit *.mp4 with a single view         -> use that exact path
    #   - explicit *.mp4 with multiple views        -> strip .mp4, append _<view>.mp4
    #   - prefix (no extension)                     -> append _<view>.mp4
    #   - omitted                                   -> save next to the NPZ
    out_arg: Path | None = args.output
    is_dir_output = False
    if out_arg is not None:
        out_str = str(out_arg)
        if out_arg.is_dir() or out_str.endswith(os.sep) or out_str.endswith("/"):
            is_dir_output = True

    if is_dir_output:
        out_dir = out_arg
        base_stem = args.npz.stem
    else:
        if out_arg is None:
            base_path = args.npz.with_suffix("")
        elif out_arg.suffix.lower() == ".mp4":
            base_path = out_arg.with_suffix("")
        else:
            base_path = out_arg
        out_dir = base_path.parent
        base_stem = base_path.name

    aspect = width / float(height)
    for view in views:
        preset = VIEW_PRESETS[view]
        fixed_cam_pose = None
        if camera_mode == "fixed":
            fixed_cam_pose = compute_fixed_camera_pose(
                preset, frame_points, marker_points, yfov=_YFOV, aspect=aspect,
            )
        print(f"[{view}] camera={camera_mode}, offset_dir={preset['offset']}")
        frames = render_motion(
            verts_all, faces, root_xz, preset, width=width, height=height,
            camera_mode=camera_mode, fixed_cam_pose=fixed_cam_pose,
            constraint_mesh=constraint_mesh,
        )
        if (
            not is_dir_output
            and len(views) == 1
            and out_arg is not None
            and out_arg.suffix.lower() == ".mp4"
        ):
            out_path = out_arg
        else:
            out_path = out_dir / f"{base_stem}_{view}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(
            str(out_path),
            np.stack(frames, axis=0),
            fps=float(args.fps),
            codec="h264",
            plugin="pyav",
        )
        print(f"  wrote {out_path}  ({len(frames)} frames @ {args.fps} fps, {width}x{height})")


if __name__ == "__main__":
    main()

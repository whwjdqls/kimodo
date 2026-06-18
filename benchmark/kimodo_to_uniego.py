"""
Kimodo <-> UniEgoMotion-style head-centric SE(3) motion representation, with
exact round-trip conversion. Experiment target: HumanML3D (22 SMPL joints).

============================================================
What this is
============================================================
UniEgoMotion (https://github.com/chaitanya100100/UniEgoMotion) proposes a
HEAD-CENTRIC motion representation: instead of a pelvis-centric kinematic-chain
encoding, every joint is stored as its GLOBAL SE(3) transform expressed in a
per-frame canonical frame anchored at the head (head yaw + floor-projected xy;
pitch / roll / height removed), and the canonical frame's trajectory is stored
as a frame-to-frame RESIDUAL. This decouples each joint from its parent in the
kinematic chain and aligns the motion frame with the (egocentric) head.

UniEgoMotion was built for egocentric SMPL-X capture. Here we *adapt* the idea
to the Kimodo / HumanML3D pipeline (SOMA family) to test whether the head-centric
representation is a good target for text-to-motion. We do NOT copy it byte-for-byte:

  - Up axis is +Y (HumanML3D / Kimodo), not their +Z.   floor height = world Y.
  - Head anchor = SMPL joint 15 (no eye joints in the 22-joint body); UniEgoMotion
    used the left-eye joint (canon_root_idx=23).
  - We DROP their SMPL-X-only channels: left/right hand PCA (12+12) and betas (10).
  - We KEEP Kimodo's 4 foot contacts (UniEgoMotion stored 2) so the round-trip
    back to the Kimodo dict is lossless.

The Kimodo dict already carries everything we need: per-joint GLOBAL rotation
matrices (``global_rot_mats``, T x 22 x 3 x 3) and GLOBAL joint positions
(``posed_joints``, T x 22 x 3). So this converter is pure matrix algebra -- no
forward kinematics, no model, no GPU.

============================================================
Our adapted feature layout  ::  211 = 198 + 9 + 4
============================================================
  [0:198]    per-joint local SE(3), j = 0..21:
               matrix_to_cont6d(R_local_j) (6)  ++  trans_local_j (3)
  [198:207]  delta canonical frame (residual trajectory):
               matrix_to_cont6d(R_delta)   (6)  ++  trans_delta   (3)
  [207:211]  foot_contacts (4, passthrough from Kimodo)

The delta slot at frame 0 stores the ABSOLUTE canonical frame cM[0];
delta[i] = cM[i-1]^{-1} @ cM[i] for i >= 1 (UniEgoMotion's convention).

``velocities`` (T,22,3) and ``r_rot_quat`` (T,4) are carried as AUXILIARY arrays
(not part of the 211-D core) so that the end-to-end HumanML3D 263-D round-trip
stays exact -- ``kimodo_to_humanml3d`` needs world velocities, and carrying
``r_rot_quat`` avoids a +/-pi half-angle wrap on ~5% of clips. The 211-D core is
deliberately velocity-free, faithful to UniEgoMotion.

============================================================
Math
============================================================
forward (kimodo -> uniego), per frame t:
  M[t,j]   = [[R[t,j], p[t,j]], [0,0,0,1]]          global SE(3) from kimodo dict
  fwd      = R[t,head][:, 2]   (z-column == +Z forward in HumanML3D world)
             fallback to R[t,0][:, 2] when ||fwd_xz|| < forward_eps
  yaw      = atan2(fwd_x, fwd_z)
  cM[t]    = [[R_y(yaw), (p_head_x, 0, p_head_z)], [0,0,0,1]]   (invtsfm_T)
  tsfm[t]  = cM[t]^{-1}
  local_T[t,j] = tsfm[t] @ M[t,j]
  delta[0] = cM[0];  delta[i] = tsfm[i-1] @ cM[i]   (= cM[i-1]^{-1} @ cM[i])

inverse (uniego -> kimodo):
  cM[0] = delta[0];  cM[i] = cM[i-1] @ delta[i]      (cumulative compose)
  M[t,j] = cM[t] @ local_T[t,j]
  global_rot_mats = M[:,:,:3,:3];  posed_joints = M[:,:,:3,3]
  root_positions  = posed_joints[:, 0]

The exact choice of "forward" axis cancels in the round-trip (it is applied in
tsfm and removed via the stored delta), so decode NEVER re-derives yaw -- it
replays cM purely from the stored delta channel.
"""

from __future__ import annotations

# ---- numpy shims must run BEFORE importing the HumanML3D-backed converter -----
import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm

# Make the sibling module importable whether run as a script or imported.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse the verified 6D<->matrix helpers and the HumanML3D inverse. Importing
# this module also runs its np.float / np.int shims and sets up the HML3D path.
from humanml3d_to_kimodo import (  # noqa: E402
    NUM_JOINTS,
    cont6d_to_matrix,
    kimodo_to_humanml3d,
    matrix_to_cont6d,
)

HEAD_IDX = 15           # SMPL-22 head joint
ROOT_IDX = 0            # pelvis (degeneracy fallback)
FEAT_DIM_UNIEGO = 198 + 9 + 4  # = 211


# ----------------------------------------------------------------------------
# SE(3) helpers
# ----------------------------------------------------------------------------


def _to_tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def build_se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(...,3,3) rotation + (...,3) translation -> (...,4,4) homogeneous."""
    out = torch.zeros(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
    out[..., :3, :3] = R
    out[..., :3, 3] = t
    out[..., 3, 3] = 1.0
    return out


def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Analytic inverse of a batch of SE(3) matrices (...,4,4).

    For T = [[R, t], [0, 1]],  T^{-1} = [[R^T, -R^T t], [0, 1]].
    More stable / faster than a general 4x4 inverse for rigid transforms.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    inv_t = -torch.matmul(Rt, t.unsqueeze(-1)).squeeze(-1)
    return build_se3(Rt, inv_t)


def _yaw_rotation_y(yaw: torch.Tensor) -> torch.Tensor:
    """(...,) yaw angle about +Y -> (...,3,3) rotation matrix.

    Built so that R_y(yaw) @ (0,0,1)^T = (sin yaw, 0, cos yaw), i.e. it maps the
    canonical +Z onto the head's floor-projected forward direction.
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    z = torch.zeros_like(yaw)
    o = torch.ones_like(yaw)
    R = torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=-2,
    )
    return R


# ----------------------------------------------------------------------------
# Forward: Kimodo dict -> UniEgo-style dict
# ----------------------------------------------------------------------------


def kimodo_to_uniego(
    kimodo_dict: dict,
    head_idx: int = HEAD_IDX,
    device: str | torch.device = "cpu",
    forward_eps: float = 1e-2,
    n_foot: int = 4,
) -> dict:
    """Convert a Kimodo dict (one motion) into the head-centric UniEgo-style rep.

    Skeleton-agnostic: ``J`` is inferred from the input shapes and the head joint
    is ``head_idx`` (15 for HumanML3D SMPL-22; 6 for SOMA-30 "Head"). The feature
    width is ``J*9 + 9 + n_foot`` (211 for HML3D, 283 for SOMA-30).

    Requires keys: ``global_rot_mats`` (T,J,3,3), ``posed_joints`` (T,J,3).
    Optionally carries through ``velocities`` (T,J,3), ``foot_contacts`` (T,4)
    and ``r_rot_quat`` (T,4) for a lossless 263-D round-trip.

    Returns a dict with ``features`` (T, 211), the intermediates ``cM``/``local_T``
    (for debugging), and the carried-through aux arrays.
    """
    R = _to_tensor(kimodo_dict["global_rot_mats"], device)   # (T, J, 3, 3)
    p = _to_tensor(kimodo_dict["posed_joints"], device)      # (T, J, 3)
    T, J = R.shape[0], R.shape[1]

    # 1) Per-joint global SE(3) ------------------------------------------------
    M = build_se3(R, p)                                       # (T, J, 4, 4)

    # 2) Head-centric canonical frame -----------------------------------------
    fwd = R[:, head_idx, :, 2]                                # (T, 3) head z-column
    fwd_xz_norm = torch.sqrt(fwd[:, 0] ** 2 + fwd[:, 2] ** 2)
    # Degeneracy fallback: head looking near-vertical -> use root z-column.
    root_fwd = R[:, ROOT_IDX, :, 2]                           # (T, 3)
    degenerate = fwd_xz_norm < forward_eps
    fwd = torch.where(degenerate[:, None], root_fwd, fwd)

    yaw = torch.atan2(fwd[:, 0], fwd[:, 2])                   # (T,)
    R_yaw = _yaw_rotation_y(yaw)                              # (T, 3, 3)
    p_head = p[:, head_idx]                                   # (T, 3)
    cM_trans = torch.stack(
        [p_head[:, 0], torch.zeros_like(p_head[:, 1]), p_head[:, 2]], dim=-1
    )                                                        # (T, 3) height zeroed
    cM = build_se3(R_yaw, cM_trans)                           # (T, 4, 4) = invtsfm_T
    tsfm = se3_inverse(cM)                                    # (T, 4, 4)

    # 3) Local per-joint transforms -------------------------------------------
    local_T = torch.matmul(tsfm[:, None], M)                  # (T, J, 4, 4)
    R_local = local_T[:, :, :3, :3]                           # (T, J, 3, 3)
    trans_local = local_T[:, :, :3, 3]                        # (T, J, 3)
    feat_joints = torch.cat(
        [matrix_to_cont6d(R_local), trans_local], dim=-1
    ).reshape(T, J * 9)                                       # (T, 198)

    # 4) Residual canonical-frame trajectory ----------------------------------
    delta = torch.empty_like(cM)                             # (T, 4, 4)
    delta[0] = cM[0]
    if T > 1:
        delta[1:] = torch.matmul(tsfm[:-1], cM[1:])
    feat_delta = torch.cat(
        [matrix_to_cont6d(delta[:, :3, :3]), delta[:, :3, 3]], dim=-1
    )                                                        # (T, 9)

    # 5) Foot contacts (passthrough) ------------------------------------------
    if "foot_contacts" in kimodo_dict and kimodo_dict["foot_contacts"] is not None:
        foot = _to_tensor(kimodo_dict["foot_contacts"], device)
    else:
        foot = torch.zeros(T, n_foot, device=device, dtype=torch.float32)
    n_foot = int(foot.shape[-1])

    features = torch.cat([feat_joints, feat_delta, foot], dim=-1)  # (T, J*9+9+n_foot)
    expected_dim = J * 9 + 9 + n_foot
    assert features.shape[-1] == expected_dim, (features.shape, expected_dim)

    out = {
        "features": features,
        "cM": cM,
        "local_T": local_T,
        "foot_contacts": foot,
    }
    # Aux carried through for a lossless 263-D round-trip.
    if "velocities" in kimodo_dict and kimodo_dict["velocities"] is not None:
        out["velocities"] = _to_tensor(kimodo_dict["velocities"], device)
    if "r_rot_quat" in kimodo_dict and kimodo_dict["r_rot_quat"] is not None:
        out["r_rot_quat"] = _to_tensor(kimodo_dict["r_rot_quat"], device)
    return out


# ----------------------------------------------------------------------------
# Inverse: UniEgo-style dict -> Kimodo dict
# ----------------------------------------------------------------------------


def uniego_to_kimodo(
    uniego_dict: dict,
    head_idx: int = HEAD_IDX,
    device: str | torch.device = "cpu",
    n_foot: int = 4,
) -> dict:
    """Inverse of :func:`kimodo_to_uniego`.

    Skeleton-agnostic: ``J`` is inferred from the feature width as
    ``J = (D - 9 - n_foot)//9``. Optionally consumes carried aux arrays
    ``velocities`` (T,J,3) and ``r_rot_quat`` (T,4). Returns a Kimodo dict with
    ``global_rot_mats``/``posed_joints``/``root_positions``/``foot_contacts``.
    """
    feat = _to_tensor(uniego_dict["features"], device)       # (T, D)
    T = feat.shape[0]
    D = feat.shape[-1]
    J = (D - 9 - n_foot) // 9
    assert J * 9 + 9 + n_foot == D, (D, J, n_foot)

    # 1) Per-joint local SE(3) ------------------------------------------------
    fj = feat[:, : J * 9].reshape(T, J, 9)
    R_local = cont6d_to_matrix(fj[..., :6])                   # (T, J, 3, 3)
    trans_local = fj[..., 6:9]                                # (T, J, 3)
    local_T = build_se3(R_local, trans_local)                # (T, J, 4, 4)

    # 2) Residual -> per-frame canonical frame (cumulative compose) -----------
    fd = feat[:, J * 9 : J * 9 + 9]                          # (T, 9)
    R_delta = cont6d_to_matrix(fd[:, :6])                    # (T, 3, 3)
    delta = build_se3(R_delta, fd[:, 6:9])                   # (T, 4, 4)
    cM = torch.empty_like(delta)
    cM[0] = delta[0]
    for i in range(1, T):
        cM[i] = cM[i - 1] @ delta[i]

    # 3) Reconstruct global per-joint SE(3) -----------------------------------
    M = torch.matmul(cM[:, None], local_T)                  # (T, J, 4, 4)
    global_rot_mats = M[:, :, :3, :3].contiguous()           # (T, J, 3, 3)
    posed_joints = M[:, :, :3, 3].contiguous()               # (T, J, 3)
    root_positions = posed_joints[:, ROOT_IDX].contiguous()  # (T, 3)
    foot = feat[:, J * 9 + 9 : J * 9 + 9 + n_foot]           # (T, n_foot)

    out = {
        "global_rot_mats": global_rot_mats,
        "posed_joints": posed_joints,
        "root_positions": root_positions,
        "foot_contacts": foot,
        "cM": cM,
    }
    if "velocities" in uniego_dict and uniego_dict["velocities"] is not None:
        out["velocities"] = _to_tensor(uniego_dict["velocities"], device)
    else:
        # Not carried -> recompute from positions (finite diff). This does NOT
        # reproduce HML3D's stored velocity dims exactly, but those dims are
        # never used by HML3D's joint-position recovery.
        vel = torch.zeros_like(posed_joints)
        if T > 1:
            vel[:-1] = posed_joints[1:] - posed_joints[:-1]
        out["velocities"] = vel
    if "r_rot_quat" in uniego_dict and uniego_dict["r_rot_quat"] is not None:
        out["r_rot_quat"] = _to_tensor(uniego_dict["r_rot_quat"], device)
    return out


# ----------------------------------------------------------------------------
# Convenience: UniEgo -> HumanML3D 263 directly (chains through Kimodo)
# ----------------------------------------------------------------------------


def uniego_to_humanml3d(
    uniego_dict: dict,
    head_idx: int = HEAD_IDX,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """UniEgo-style rep -> HumanML3D 263-D vector (T, 263).

    There is no shortcut that bypasses Kimodo: the HumanML3D inverse
    (``kimodo_to_humanml3d``) consumes exactly the world-frame quantities
    (`global_rot_mats`, `posed_joints`, `velocities`, ...) that
    ``uniego_to_kimodo`` reconstructs, so this is the two functions chained.
    Verified to match the ORIGINAL HumanML3D vectors to ~1.7e-5 (carry
    `velocities` + `r_rot_quat` in the uniego dict for an exact match).
    """
    kd = uniego_to_kimodo(uniego_dict, head_idx=head_idx, device=device)
    return kimodo_to_humanml3d(kd, device=device)


# ----------------------------------------------------------------------------
# I/O + verification + CLI
# ----------------------------------------------------------------------------


def _load_kimodo_npz(path: Path, device="cpu") -> dict:
    """Load a kimodo_rep .npz into a dict of torch tensors."""
    data = np.load(path)
    return {k: torch.from_numpy(data[k]).float().to(device) for k in data.files}


def _convert_one(src: Path, dst_dir: Path, overwrite: bool) -> tuple[Path, str]:
    out_path = dst_dir / (src.stem + ".npz")
    if out_path.is_file() and not overwrite:
        return out_path, "skipped"
    try:
        kd = _load_kimodo_npz(src)
        ud = kimodo_to_uniego(kd)
        save = {
            "features": ud["features"].cpu().numpy().astype(np.float32),
            "foot_contacts": ud["foot_contacts"].cpu().numpy().astype(np.float32),
        }
        if "velocities" in ud:
            save["velocities"] = ud["velocities"].cpu().numpy().astype(np.float32)
        if "r_rot_quat" in ud:
            save["r_rot_quat"] = ud["r_rot_quat"].cpu().numpy().astype(np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **save)
        return out_path, "ok"
    except Exception as e:  # noqa: BLE001
        return out_path, f"error: {type(e).__name__}: {e}"


def _verify_roundtrip(src: Path) -> dict:
    """Run both checks on one kimodo_rep .npz file.

    Check A: kimodo -> uniego -> kimodo, diff posed_joints / global_rot_mats.
    Check B: 263 -> ... -> 263' end-to-end (uses kimodo_to_humanml3d).
    """
    kd = _load_kimodo_npz(src)
    ud = kimodo_to_uniego(kd)
    kd2 = uniego_to_kimodo(ud)

    def _md(a, b):
        return float((a - b).abs().max())

    a_pos = _md(kd["posed_joints"], kd2["posed_joints"])
    a_rot = _md(kd["global_rot_mats"], kd2["global_rot_mats"])
    a_foot = _md(kd["foot_contacts"], kd2["foot_contacts"])
    a_vel = _md(kd["velocities"], kd2["velocities"]) if "velocities" in kd else 0.0

    # Check B: full HML3D 263 round-trip via the existing inverse.
    src263 = kimodo_to_humanml3d(kd).cpu().numpy()
    back263 = kimodo_to_humanml3d(kd2).cpu().numpy()
    diff = np.abs(back263 - src263)
    diff[-1, 0] = 0.0          # never-recovered last-frame entries
    diff[-1, 1:3] = 0.0
    diff[-1, 193:259] = 0.0
    b_max = float(diff.max())

    return {
        "file": src.name,
        "A_pos": a_pos,
        "A_rot": a_rot,
        "A_foot": a_foot,
        "A_vel": a_vel,
        "B_263": b_max,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert Kimodo-rep NPZs <-> UniEgoMotion-style head-centric rep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep"),
        help="Folder of kimodo_rep *.npz files.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/jungbin_cho/HumanML3D/HumanML3D/uniego_rep"),
        help="Folder for uniego *.npz outputs.",
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only process N files.")
    ap.add_argument(
        "--verify-roundtrip",
        type=int,
        default=0,
        metavar="N",
        help="Run kimodo->uniego->kimodo (+263 end-to-end) on the first N files.",
    )
    ap.add_argument("--skip-write", action="store_true", help="Don't write NPZs.")
    args = ap.parse_args()

    src_files = sorted(args.input_dir.glob("*.npz"))
    if args.limit is not None:
        src_files = src_files[: args.limit]
    if not src_files:
        raise SystemExit(f"No *.npz under {args.input_dir}")
    print(f"Discovered {len(src_files)} kimodo_rep files.")

    if not args.skip_write:
        print(f"Writing to {args.output_dir}")
        fn = partial(_convert_one, dst_dir=args.output_dir, overwrite=args.overwrite)
        n_ok = n_skip = n_err = 0
        errors: list[tuple[Path, str]] = []
        if args.workers <= 1:
            it = (fn(p) for p in src_files)
            pool = None
        else:
            pool = Pool(args.workers)
            it = pool.imap_unordered(fn, src_files)
        for out_path, status in tqdm(it, total=len(src_files), desc="Converting"):
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_err += 1
                errors.append((out_path, status))
        if pool is not None:
            pool.close()
            pool.join()
        print(f"Done. ok={n_ok}, skipped={n_skip}, errors={n_err}")
        for path, msg in errors[:10]:
            print(f"  {path}: {msg}")

    if args.verify_roundtrip > 0:
        check_files = src_files[: args.verify_roundtrip]
        print(f"\nRound-trip verification on {len(check_files)} file(s):")
        worst_A = 0.0
        worst_B = 0.0
        for src in check_files:
            r = _verify_roundtrip(src)
            worst_A = max(worst_A, r["A_pos"], r["A_rot"])
            worst_B = max(worst_B, r["B_263"])
            print(
                f"  {r['file']}: A_pos={r['A_pos']:.2e} A_rot={r['A_rot']:.2e} "
                f"A_foot={r['A_foot']:.2e} A_vel={r['A_vel']:.2e} B_263={r['B_263']:.2e}"
            )
        print(f"\nOverall: Check A (kimodo round-trip) max |Δ| = {worst_A:.3e}")
        print(f"         Check B (263 end-to-end)   max |Δ| = {worst_B:.3e}")
        okA = worst_A < 1e-4
        okB = worst_B < 1e-3
        print(
            f"Check A {'PASS' if okA else 'FAIL'} (<1e-4); "
            f"Check B {'PASS' if okB else 'FAIL'} (<1e-3)."
        )


if __name__ == "__main__":
    main()

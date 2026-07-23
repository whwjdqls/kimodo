"""
NymeriaPlus (proportional, shape-aware) kimodo motion -> UniEgoMotion-style
head-centric SE(3) rep (283-D, SOMA-30).

This is the Nymeria counterpart to ``benchmark/soma_to_uniego.py``. The difference
is the *source schema*: the proportional NymeriaPlus NPZs store **local** joint
rotations + a per-actor rest skeleton, not the precomputed global SE(3) that the
BONES-SEED raw SOMA NPZs carry. So we run FK first.

Source NPZ ``/weka/jungbin/nymeriaplus_kimodo_proportional/{Sxx}/{seq}.npz`` keys:
  local_rot_mats (T,77,3,3), root_positions (T,3), neutral_joints (77,3)  [per-actor],
  identity_coeffs (1,10), floor_offset (), grounded (bool), timestamps_us (T,), fps.

Pipeline (mirrors the two-hop chain in UNIEGO_REPRESENTATION.md, FK-direct variant):
  1. FK (77)  : SOMASkeleton77.fk(local_rot_mats, root, neutral_joints)
                -> global_rot_mats (T,77,3,3), posed_joints (T,77,3).   PER-ACTOR shape
                is preserved because FK runs on this actor's neutral_joints.
  2. slice    : 77 -> SOMA-30 subset (same name map the kimodo dataset uses).
  3. feet     : foot contacts vs a PER-FRAME LOCAL FLOOR (sliding low-percentile of
                foot height) + a velocity gate -> foot_contacts (T,4). Grounding- and
                multi-floor-independent (the rep's plain foot_detect needs one global
                floor, which is wrong here and zeroed ~40% of contacts).
  4. uniego   : kimodo_to_uniego(head_idx=6, n_foot=4) -> features (T,283).

GROUNDING: by default positions are kept RAW (grounded=False, the proportional NPZ's
own convention -- raw SLAM-world height). Downstream windowed training grounds each
window by the manifest ``ground_offset_y`` exactly like the motion + camera streams
(subtract it from the per-joint trans Y block; rotations/canonical-delta unaffected).
Pass ``--ground`` to instead bake whole-seq ``floor_offset`` (lowest-floor) grounding.

Output ``uniego_rep/{Sxx}/{seq}.npz``:
  features (T,283), foot_contacts (T,4), neutral_joints (30,3) [SOMA-30, per-actor,
  for downstream ShapeEncoder conditioning], identity_coeffs (1,10), floor_offset (),
  grounded (bool), fps, timestamps_us (T,)  [carried so the uniego rep stays frame-
  aligned 1:1 with the ego video / camera / motion windows].

The 283-D rep encodes per-actor bone lengths implicitly (joint positions are global),
so it is shape-aware for free; see UNIEGO_REPRESENTATION.md sec 14 for the bone-length
caveat.

Round-trip precision: PER-WINDOW it is float32-exact (~2-5e-5) -- which is how the rep
is consumed (each training window re-canonicalizes its own frame 0 per UNIEGO sec 10,
so decode composes the residual canonical frame cM over <= a few hundred frames). A
WHOLE-SEQUENCE decode cumulatively composes cM over all T frames; on these ~16-min
(~19k-frame) Nymeria clips float32 rounding drifts to ~1-2 mm and grows with T. That
is an inherent property of the residual-trajectory encoding on very long sequences, NOT
a converter bug (BONES-SEED clips at T<=201 hit 2e-5), and is physically negligible
(below the SMPL fit's ~1.6 mm/frame foot jitter). --verify-roundtrip reports both.

Usage:
  python benchmark/nymeria_to_uniego.py --verify-roundtrip 12 --skip-write
  python benchmark/nymeria_to_uniego.py --workers 16        # convert all 732
Env: kimodo. CPU-only.
"""
from __future__ import annotations

import numpy as np

if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

import argparse
import sys
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
for _p in (str(_THIS_DIR), str(_THIS_DIR.parent)):   # benchmark/ + repo root (for `import kimodo`)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kimodo_to_uniego import kimodo_to_uniego, uniego_to_kimodo  # noqa: E402

NY_HEAD_IDX = 6      # "Head" in SOMASkeleton30
NY_N_JOINTS = 30
NY_N_FOOT = 4
NY_FEAT_DIM = NY_N_JOINTS * 9 + 9 + NY_N_FOOT  # 283
FPS = 20.0

MROOT = Path("/weka/jungbin/nymeriaplus_kimodo_proportional")


@lru_cache(maxsize=1)
def _idx_77_to_30() -> tuple:
    from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
    s30, s77 = SOMASkeleton30(), SOMASkeleton77()
    return tuple(s77.bone_index[n] for n in s30.bone_order_names)


@lru_cache(maxsize=1)
def _skels():
    from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
    return SOMASkeleton77(), SOMASkeleton30()


def _robust_foot_contacts(posed30, s30, fps, win=20, pct=10.0,
                          vel_thres=0.15, height_thresh=0.10):
    """Foot contacts robust to grounding + multi-floor.

    The rep's own ``foot_detect_from_pos_and_vel`` gates on absolute foot height
    (< height_thresh above Y=0), so it only works if the body is grounded to the
    floor the foot is standing on. On these raw / multi-floor Nymeria sequences a
    single global floor is wrong (whole-seq min-foot grounding floats the body and
    kills ~40% of contacts). Instead we estimate a **per-frame local floor** as the
    ``pct``-percentile of the min foot height over a +-``win``-frame window, and gate
    contact on (foot within height_thresh of that local floor) AND (foot speed <
    vel_thres). Same per-side foot joints + thresholds as the rep, just a sliding
    floor — so it adapts to stairs/raised surfaces and is grounding-independent.
    Returns (T,4) in [L_heel, L_toe, R_heel, R_toe] order.
    """
    from kimodo.motion_rep.feature_utils import compute_vel_xyz
    fid = list(s30.left_foot_joint_idx[:2]) + list(s30.right_foot_joint_idx[:2])
    h = posed30[:, fid, 1]                                  # (T,4) foot heights
    minh = h.min(dim=1).values.detach().cpu().numpy()       # (T,)
    T = minh.shape[0]
    padded = np.pad(minh, win, mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(padded, 2 * win + 1)   # (T, 2win+1)
    local_floor = torch.from_numpy(np.percentile(sw, pct, axis=1)).to(posed30)  # (T,)
    vel = compute_vel_xyz(posed30, fps)                    # (T,30,3)
    fvel = torch.linalg.norm(vel[:, fid], dim=-1)          # (T,4)
    above = h - local_floor[:, None]                       # (T,4) height over local floor
    contact = ((above < height_thresh) & (fvel < vel_thres)).float()
    return contact


def nymeria_npz_to_kimodo_dict(path: Path, ground: bool = False) -> dict:
    """proportional NymeriaPlus NPZ -> kimodo dict (SOMA-30 global SE(3) + feet).

    ``ground=False`` (default): keep RAW SLAM-world height (the proportional NPZ's
    own convention, ``grounded=False``), so downstream windowed training grounds each
    window by the manifest ``ground_offset_y`` exactly like the motion + camera. Pass
    ``ground=True`` to bake whole-seq ``floor_offset`` grounding (lowest floor only).
    Foot contacts are computed grounding-independently (per-frame local floor), so
    they are correct either way.

    Returns ``global_rot_mats`` (T,30,3,3), ``posed_joints`` (T,30,3),
    ``foot_contacts`` (T,4), plus carried ``neutral_joints`` (30,3),
    ``identity_coeffs``, ``floor_offset``, ``timestamps_us``.
    """
    s77, s30 = _skels()
    idx = list(_idx_77_to_30())
    d = np.load(path, allow_pickle=True)

    lrm = torch.from_numpy(d["local_rot_mats"].astype(np.float32))   # (T,77,3,3)
    root = torch.from_numpy(d["root_positions"].astype(np.float32))  # (T,3)
    nj77 = torch.from_numpy(d["neutral_joints"].astype(np.float32))  # (77,3)
    floor_offset = float(d["floor_offset"]) if "floor_offset" in d.files else 0.0
    T = lrm.shape[0]

    if ground:
        root = root.clone()
        root[:, 1] -= floor_offset                                   # feet -> ~0 (lowest floor)

    # FK on this actor's rest skeleton (frames as the batch dim).
    nj_b = nj77.unsqueeze(0).expand(T, -1, -1)                       # (T,77,3)
    global_rots77, posed77, _ = s77.fk(lrm, root, neutral_joints=nj_b)
    global30 = global_rots77[:, idx].contiguous()                   # (T,30,3,3)
    posed30 = posed77[:, idx].contiguous()                          # (T,30,3)

    foot = _robust_foot_contacts(posed30, s30, FPS)                # (T,4) grounding-independent

    return {
        "global_rot_mats": global30,
        "posed_joints": posed30,
        "foot_contacts": foot,
        "neutral_joints30": nj77[idx].contiguous(),                # (30,3) per-actor
        "identity_coeffs": d["identity_coeffs"] if "identity_coeffs" in d.files else None,
        "floor_offset": floor_offset,
        "grounded": bool(ground),
        "timestamps_us": d["timestamps_us"] if "timestamps_us" in d.files else None,
    }


def nymeria_npz_to_uniego(path: Path, ground: bool = False) -> dict:
    kd = nymeria_npz_to_kimodo_dict(path, ground=ground)
    ud = kimodo_to_uniego(kd, head_idx=NY_HEAD_IDX, n_foot=NY_N_FOOT)
    ud["neutral_joints30"] = kd["neutral_joints30"]
    ud["identity_coeffs"] = kd["identity_coeffs"]
    ud["floor_offset"] = kd["floor_offset"]
    ud["grounded"] = kd["grounded"]
    ud["timestamps_us"] = kd["timestamps_us"]
    return ud


# ---------------------------------------------------------------------------
# I/O + verification + CLI
# ---------------------------------------------------------------------------
def _convert_one(src: Path, in_root: Path, dst_dir: Path, overwrite: bool, ground: bool):
    rel = src.relative_to(in_root)
    out_path = dst_dir / rel
    if out_path.is_file() and not overwrite:
        return out_path, "skipped"
    try:
        ud = nymeria_npz_to_uniego(src, ground=ground)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save = {
            "features": ud["features"].cpu().numpy().astype(np.float32),
            "foot_contacts": ud["foot_contacts"].cpu().numpy().astype(np.float32),
            "neutral_joints": ud["neutral_joints30"].cpu().numpy().astype(np.float32),
            "floor_offset": np.float32(ud["floor_offset"]),
            "grounded": np.bool_(ud["grounded"]),
            "fps": np.int64(FPS),
        }
        if ud["identity_coeffs"] is not None:
            save["identity_coeffs"] = np.asarray(ud["identity_coeffs"], np.float32)
        if ud["timestamps_us"] is not None:
            save["timestamps_us"] = np.asarray(ud["timestamps_us"], np.int64)
        np.savez(out_path, **save)
        return out_path, "ok"
    except Exception as e:  # noqa: BLE001
        return out_path, f"error: {type(e).__name__}: {e}"


def _verify_roundtrip(src: Path, ground: bool, window: int = 200) -> dict:
    """Round-trip both ways.

    ``A_*_win`` is the PER-WINDOW metric (the rep as it is actually consumed: each
    training window re-canonicalizes its own frame 0, so decode only composes the
    residual ``cM`` over <= ``window`` frames). This is the meaningful number and is
    float32-exact (~2-5e-5). ``A_pos_full`` is the WHOLE-SEQUENCE decode, which
    cumulatively composes ``cM`` over all T (~19k) frames; float32 rounding there
    drifts to ~mm and grows with T -- an inherent property of the residual rep on
    very long clips, NOT a converter error, and physically negligible (< SMPL jitter).
    """
    def _md(a, b):
        return float((a - b).abs().max())

    kd = nymeria_npz_to_kimodo_dict(src, ground=ground)
    T = int(kd["posed_joints"].shape[0])
    dim_ref = [0]

    # whole-sequence (cumulative) round-trip -- informational
    ud = kimodo_to_uniego(kd, head_idx=NY_HEAD_IDX, n_foot=NY_N_FOOT)
    kd2 = uniego_to_kimodo(ud, head_idx=NY_HEAD_IDX, n_foot=NY_N_FOOT)
    A_pos_full = _md(kd["posed_joints"], kd2["posed_joints"])

    # per-window round-trip -- the metric that matches training use
    wp = wr = wf = 0.0
    for s in range(0, T, window):
        e = min(s + window, T)
        sub = {k: v[s:e] for k, v in kd.items()
               if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == T}
        uw = kimodo_to_uniego(sub, head_idx=NY_HEAD_IDX, n_foot=NY_N_FOOT)
        kw = uniego_to_kimodo(uw, head_idx=NY_HEAD_IDX, n_foot=NY_N_FOOT)
        wp = max(wp, _md(sub["posed_joints"], kw["posed_joints"]))
        wr = max(wr, _md(sub["global_rot_mats"], kw["global_rot_mats"]))
        wf = max(wf, _md(sub["foot_contacts"], kw["foot_contacts"]))
        dim_ref[0] = int(uw["features"].shape[-1])

    return {
        "file": f"{src.parent.name}/{src.name}", "T": T, "dim": dim_ref[0],
        "A_pos_win": wp, "A_rot_win": wr, "A_foot_win": wf,
        "A_pos_full": A_pos_full,
        "contact_frac": float(kd["foot_contacts"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=MROOT,
                    help="Root with {Sxx}/{seq}.npz proportional motion NPZs.")
    ap.add_argument("--output-dir", type=Path, default=MROOT / "uniego_rep")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ground", action="store_true",
                    help="Bake whole-seq floor_offset grounding into stored positions "
                         "(default: keep RAW height; ground per-window downstream).")
    ap.add_argument("--verify-roundtrip", type=int, default=0, metavar="N")
    ap.add_argument("--skip-write", action="store_true")
    args = ap.parse_args()
    ground = args.ground

    # exactly the top-level motion NPZs ({Sxx}/{seq}.npz) -- NOT camera/ video/ etc.
    src_files = sorted(args.input_dir.glob("S*/*.npz"))
    if args.limit is not None:
        src_files = src_files[: args.limit]
    if not src_files:
        raise SystemExit(f"No S*/*.npz under {args.input_dir}")
    print(f"Discovered {len(src_files)} NymeriaPlus motion files. "
          f"(J=30, head=6, dim={NY_FEAT_DIM}, ground={ground})")

    if not args.skip_write:
        print(f"Writing to {args.output_dir}")
        fn = partial(_convert_one, in_root=args.input_dir, dst_dir=args.output_dir,
                     overwrite=args.overwrite, ground=ground)
        n_ok = n_skip = n_err = 0
        errors = []
        if args.workers <= 1:
            it, pool = (fn(p) for p in src_files), None
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
            pool.close(); pool.join()
        print(f"Done. ok={n_ok}, skipped={n_skip}, errors={n_err}")
        for path, msg in errors[:10]:
            print(f"  {path}: {msg}")

    if args.verify_roundtrip > 0:
        check_files = src_files[: args.verify_roundtrip]
        print(f"\nNymeria round-trip verification on {len(check_files)} file(s):")
        worst_win = worst_full = 0.0
        for src in check_files:
            r = _verify_roundtrip(src, ground=ground)
            worst_win = max(worst_win, r["A_pos_win"], r["A_rot_win"])
            worst_full = max(worst_full, r["A_pos_full"])
            print(f"  {r['file']}: T={r['T']} dim={r['dim']} | "
                  f"WIN A_pos={r['A_pos_win']:.2e} A_rot={r['A_rot_win']:.2e} "
                  f"A_foot={r['A_foot_win']:.2e} | full A_pos={r['A_pos_full']:.2e} "
                  f"| contact_frac={r['contact_frac']:.2f}")
        print(f"\nPer-window (training-use)  max |Δ| = {worst_win:.3e}  -> "
              f"{'PASS' if worst_win < 1e-4 else 'FAIL'} (<1e-4)")
        print(f"Whole-sequence (cumulative) max |Δ| = {worst_full:.3e}  "
              f"(informational; ~mm drift over ~19k frames is inherent + negligible)")


if __name__ == "__main__":
    main()

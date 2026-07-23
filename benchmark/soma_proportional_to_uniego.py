"""
SHAPE-AWARE BONES-SEED (proportional) kimodo rep -> UniEgoMotion head-centric rep.

Proportional counterpart to ``benchmark/soma_to_uniego.py`` (which targets the
shape-FREE uniform set). The proportional motions at
``/weka/jungbin/seed/soma_proportional_motions_20fps/{date}/{name}.npz`` were FK'd on
each actor's own rest skeleton, so they carry, in addition to the usual SOMA keys,
a per-actor ``neutral_joints (77,3)``. Because ``posed_joints`` are already FK'd on
that per-actor skeleton, the resulting uniego rep encodes per-actor bone lengths --
i.e. it is **shape-aware for free** (UNIEGO_REPRESENTATION.md sec 14). We additionally
**carry the per-actor neutral_joints (sliced to SOMA-30)** so a downstream ShapeEncoder
can condition on the target body.

Difference vs the raw/Nymeria path:
  * The proportional NPZs already store precomputed ``global_rot_mats``, ``posed_joints``
    and ``foot_contacts`` -- no FK and no foot re-detection needed (unlike
    nymeria_to_uniego, which FKs from local rotations and recomputes contacts).
  * The BVH source is **floor-referenced** (min foot Y ~ +0.01 m), so positions are
    grounded as-is and the precomputed foot_contacts are valid (contact_frac ~0.9 on
    jump/land clips) -- pass them straight through.
  * Clips are short (T <~ 400), so the residual-canonical round-trip is fully
    float32-exact (~1-2e-5); no long-sequence cumulative drift (cf. Nymeria's 16-min
    clips).

Params: J=30, head_idx=6 ("Head"), n_foot=4, feature dim 283. Up axis +Y.

Output ``soma_proportional_uniegomotion_20fps/{date}/{name}.npz``:
  features (T,283), foot_contacts (T,4), neutral_joints (30,3) [SOMA-30, per-actor].

Usage:
  python benchmark/soma_proportional_to_uniego.py --verify-roundtrip 12 --skip-write
  python benchmark/soma_proportional_to_uniego.py --workers 32      # convert all
Env: kimodo. CPU-only (ssh into an a3ultra node; do NOT srun a GPU node for CPU work).
"""
from __future__ import annotations

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

_THIS_DIR = Path(__file__).resolve().parent
for _p in (str(_THIS_DIR), str(_THIS_DIR.parent)):   # benchmark/ + repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kimodo_to_uniego import kimodo_to_uniego, uniego_to_kimodo  # noqa: E402
from soma_to_uniego import _idx_77_to_30, SOMA_HEAD_IDX, SOMA_N_FOOT, SOMA_FEAT_DIM  # noqa: E402

DEFAULT_IN = Path("/weka/jungbin/seed/soma_proportional_motions_20fps")
DEFAULT_OUT = Path("/weka/jungbin/seed/soma_proportional_uniegomotion_20fps")


def _prop_dict_from_npz(path: Path) -> dict:
    """proportional SOMA .npz -> kimodo dict (SOMA-30 global/posed/feet + per-actor
    neutral_joints), all sliced 77->30. Uses the PRECOMPUTED global_rot_mats /
    posed_joints / foot_contacts (no FK)."""
    idx = list(_idx_77_to_30())
    d = np.load(path, allow_pickle=True)
    grm = np.asarray(d["global_rot_mats"])[:, idx]   # (T,30,3,3)
    pj = np.asarray(d["posed_joints"])[:, idx]       # (T,30,3)
    fc = np.asarray(d["foot_contacts"])              # (T,4)
    nj = np.asarray(d["neutral_joints"])[idx]        # (30,3) per-actor
    return {
        "global_rot_mats": torch.from_numpy(grm).float(),
        "posed_joints": torch.from_numpy(pj).float(),
        "foot_contacts": torch.from_numpy(fc).float(),
        "neutral_joints30": torch.from_numpy(nj).float(),
    }


def prop_npz_to_uniego(path: Path) -> dict:
    kd = _prop_dict_from_npz(path)
    ud = kimodo_to_uniego(kd, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)
    ud["neutral_joints30"] = kd["neutral_joints30"]
    return ud


# ---------------------------------------------------------------------------
def _convert_one(src: Path, in_root: Path, dst_dir: Path, overwrite: bool):
    rel = src.relative_to(in_root)
    out_path = dst_dir / rel
    if out_path.is_file() and not overwrite:
        return out_path, "skipped"
    try:
        ud = prop_npz_to_uniego(src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            features=ud["features"].cpu().numpy().astype(np.float32),
            foot_contacts=ud["foot_contacts"].cpu().numpy().astype(np.float32),
            neutral_joints=ud["neutral_joints30"].cpu().numpy().astype(np.float32),
        )
        return out_path, "ok"
    except Exception as e:  # noqa: BLE001
        return out_path, f"error: {type(e).__name__}: {e}"


def _verify_roundtrip(src: Path) -> dict:
    kd = _prop_dict_from_npz(src)
    ud = kimodo_to_uniego(kd, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)
    kd2 = uniego_to_kimodo(ud, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)

    def _md(a, b):
        return float((a - b).abs().max())

    return {
        "file": f"{src.parent.name}/{src.name}",
        "T": int(kd["posed_joints"].shape[0]),
        "dim": int(ud["features"].shape[-1]),
        "A_pos": _md(kd["posed_joints"], kd2["posed_joints"]),
        "A_rot": _md(kd["global_rot_mats"], kd2["global_rot_mats"]),
        "A_foot": _md(kd["foot_contacts"], kd2["foot_contacts"]),
        "contact_frac": float(kd["foot_contacts"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-roundtrip", type=int, default=0, metavar="N")
    ap.add_argument("--skip-write", action="store_true")
    args = ap.parse_args()

    src_files = sorted(args.input_dir.rglob("*.npz"))
    if args.limit is not None:
        src_files = src_files[: args.limit]
    if not src_files:
        raise SystemExit(f"No *.npz under {args.input_dir}")
    print(f"Discovered {len(src_files)} proportional SOMA files. "
          f"(J=30, head=6, dim={SOMA_FEAT_DIM}, shape-aware + neutral_joints carried)")

    if not args.skip_write:
        print(f"Writing to {args.output_dir}")
        fn = partial(_convert_one, in_root=args.input_dir, dst_dir=args.output_dir,
                     overwrite=args.overwrite)
        n_ok = n_skip = n_err = 0
        errors = []
        if args.workers <= 1:
            it, pool = (fn(p) for p in src_files), None
        else:
            pool = Pool(args.workers)
            it = pool.imap_unordered(fn, src_files, chunksize=16)
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
        print(f"\nProportional round-trip verification on {len(check_files)} file(s):")
        worst = 0.0
        for src in check_files:
            r = _verify_roundtrip(src)
            worst = max(worst, r["A_pos"], r["A_rot"])
            print(f"  {r['file']}: T={r['T']} dim={r['dim']} "
                  f"A_pos={r['A_pos']:.2e} A_rot={r['A_rot']:.2e} "
                  f"A_foot={r['A_foot']:.2e} contact_frac={r['contact_frac']:.2f}")
        print(f"\nOverall Check A max |Δ| = {worst:.3e}  -> "
              f"{'PASS' if worst < 1e-4 else 'FAIL'} (<1e-4)")


if __name__ == "__main__":
    main()

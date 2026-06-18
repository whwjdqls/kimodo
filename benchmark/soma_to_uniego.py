"""
SOMA (BONES-SEED / Nymeria) kimodo rep -> UniEgoMotion-style head-centric rep.

SOMA counterpart to ``benchmark/kimodo_to_uniego.py``'s HumanML3D CLI. The
head-centric SE(3) conversion itself is skeleton-agnostic (see
``kimodo_to_uniego`` / ``uniego_to_kimodo``); this driver just supplies the SOMA
ingredients:

  * Source: raw SOMA motion ``.npz`` (77-joint) which already store
    ``global_rot_mats (T,77,3,3)``, ``posed_joints (T,77,3)``, ``foot_contacts
    (T,4)``. We slice the 77-joint arrays to the **SOMA-30** subset used by the
    kimodo rep (the same 77->30 mapping the dataset uses), giving the global
    SE(3) ingredients per joint.
  * Params: ``J=30``, ``head_idx=6`` ("Head"), ``n_foot=4``. Feature width
    ``30*9 + 9 + 4 = 283``. Up axis +Y, same 6D convention as HumanML3D.

Output ``.npz`` per clip: ``features (T,283)`` + ``foot_contacts (T,4)``.

NOTE on body shape: the uniform SOMA tree is built on a single canonical
skeleton, so the joint positions (hence bone lengths) encoded here are canonical.
For per-actor shape, source ``posed_joints`` from the proportional pipeline
(per-actor ``neutral_joints``) — the converter preserves whatever shape the input
positions carry. See benchmark/UNIEGO_REPRESENTATION.md (shape-awareness section).

Usage:
  python benchmark/soma_to_uniego.py --verify-roundtrip 12 --skip-write
  python benchmark/soma_to_uniego.py --input-dir <raw_soma> --output-dir <uniego> --workers 8
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
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from kimodo_to_uniego import kimodo_to_uniego, uniego_to_kimodo  # noqa: E402

SOMA_HEAD_IDX = 6   # "Head" in SOMASkeleton30
SOMA_N_JOINTS = 30
SOMA_N_FOOT = 4
SOMA_FEAT_DIM = SOMA_N_JOINTS * 9 + 9 + SOMA_N_FOOT  # 283


@lru_cache(maxsize=1)
def _idx_77_to_30() -> tuple:
    """The 77->30 joint index map (by bone name), matching the kimodo dataset's
    SOMASkeleton30 subset of SOMASkeleton77."""
    from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
    s30, s77 = SOMASkeleton30(), SOMASkeleton77()
    return tuple(s77.bone_index[n] for n in s30.bone_order_names)


def _soma_dict_from_npz(path: Path) -> dict:
    """Load a raw SOMA .npz and slice 77->30 into the kimodo-dict the converter
    needs (``global_rot_mats`` (T,30,3,3), ``posed_joints`` (T,30,3),
    ``foot_contacts`` (T,4))."""
    idx = list(_idx_77_to_30())
    d = np.load(path)
    grm = np.asarray(d["global_rot_mats"])[:, idx]   # (T,30,3,3)
    pj = np.asarray(d["posed_joints"])[:, idx]       # (T,30,3)
    fc = np.asarray(d["foot_contacts"])              # (T,4)
    return {
        "global_rot_mats": torch.from_numpy(grm).float(),
        "posed_joints": torch.from_numpy(pj).float(),
        "foot_contacts": torch.from_numpy(fc).float(),
    }


def soma_npz_to_uniego(path: Path) -> dict:
    """Raw SOMA .npz -> uniego dict (features (T,283) + foot_contacts)."""
    kd = _soma_dict_from_npz(path)
    return kimodo_to_uniego(kd, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)


# ----------------------------------------------------------------------------
# I/O + verification + CLI
# ----------------------------------------------------------------------------
def _convert_one(src: Path, in_root: Path, dst_dir: Path, overwrite: bool):
    rel = src.relative_to(in_root).with_suffix(".npz")
    out_path = dst_dir / rel
    if out_path.is_file() and not overwrite:
        return out_path, "skipped"
    try:
        ud = soma_npz_to_uniego(src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            features=ud["features"].cpu().numpy().astype(np.float32),
            foot_contacts=ud["foot_contacts"].cpu().numpy().astype(np.float32),
        )
        return out_path, "ok"
    except Exception as e:  # noqa: BLE001
        return out_path, f"error: {type(e).__name__}: {e}"


def _verify_roundtrip(src: Path) -> dict:
    """SOMA round-trip Check A: kimodo(30) -> uniego -> kimodo, diff arrays."""
    kd = _soma_dict_from_npz(src)
    ud = kimodo_to_uniego(kd, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)
    kd2 = uniego_to_kimodo(ud, head_idx=SOMA_HEAD_IDX, n_foot=SOMA_N_FOOT)

    def _md(a, b):
        return float((a - b).abs().max())

    return {
        "file": src.name,
        "T": int(kd["posed_joints"].shape[0]),
        "dim": int(ud["features"].shape[-1]),
        "A_pos": _md(kd["posed_joints"], kd2["posed_joints"]),
        "A_rot": _md(kd["global_rot_mats"], kd2["global_rot_mats"]),
        "A_foot": _md(kd["foot_contacts"], kd2["foot_contacts"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input-dir", type=Path,
                    default=Path("/home/jungbin_cho/seed/soma_uniform_motions_20fps"),
                    help="Root of raw SOMA *.npz motions (searched recursively).")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/home/jungbin_cho/seed/soma_uniego_rep_20fps"))
    ap.add_argument("--workers", type=int, default=1)
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
    print(f"Discovered {len(src_files)} SOMA motion files. (J=30, head=6, dim={SOMA_FEAT_DIM})")

    if not args.skip_write:
        print(f"Writing to {args.output_dir}")
        fn = partial(_convert_one, in_root=args.input_dir,
                     dst_dir=args.output_dir, overwrite=args.overwrite)
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
        print(f"\nSOMA round-trip verification on {len(check_files)} file(s):")
        worst = 0.0
        for src in check_files:
            r = _verify_roundtrip(src)
            worst = max(worst, r["A_pos"], r["A_rot"])
            print(f"  {r['file']}: T={r['T']} dim={r['dim']} "
                  f"A_pos={r['A_pos']:.2e} A_rot={r['A_rot']:.2e} A_foot={r['A_foot']:.2e}")
        print(f"\nOverall Check A max |Δ| = {worst:.3e}  -> "
              f"{'PASS' if worst < 1e-4 else 'FAIL'} (<1e-4)")


if __name__ == "__main__":
    main()

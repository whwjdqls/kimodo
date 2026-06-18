"""
Side-by-side visualization of the Kimodo <-> UniEgoMotion-style round-trip.

For each requested HumanML3D clip we load its kimodo_rep, run
``kimodo_to_uniego`` then ``uniego_to_kimodo``, and render the ORIGINAL joints
(left, blue) against the ROUND-TRIPPED joints (right, red) with the per-clip
max position error in the caption. Because the representation is exactly
invertible (~1e-5 m), the two skeletons should be visually identical -- the
caption is what makes the check legible.

Usage:
  python benchmark/viz_uniego_roundtrip.py                 # default 6 clips
  python benchmark/viz_uniego_roundtrip.py --ids 000004 000006 M000123
  python benchmark/viz_uniego_roundtrip.py --n 10 --fps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from kimodo_to_uniego import (  # noqa: E402
    _load_kimodo_npz,
    kimodo_to_uniego,
    uniego_to_kimodo,
)

from kimodo.scripts.render_hml3d import render_sidebyside  # noqa: E402

KIMODO_REP = Path("/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep")
TEXTS_DIR = Path("/home/jungbin_cho/HumanML3D/HumanML3D/texts")
DEFAULT_IDS = ["000000", "000004", "000006", "000010", "000021", "000050"]


def _caption_for(clip_id: str) -> str:
    """First text annotation for a clip (HML3D 'caption#tokens#...' format)."""
    p = TEXTS_DIR / f"{clip_id}.txt"
    if not p.is_file():
        return clip_id
    try:
        line = p.read_text().strip().splitlines()[0]
        return line.split("#")[0].strip()
    except Exception:  # noqa: BLE001
        return clip_id


def _save(frames: np.ndarray, out: Path, fps: float) -> Path:
    """Write frames to MP4 (h264 via pyav); fall back to GIF on failure."""
    import imageio.v3 as iio

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(out, frames, fps=fps, codec="h264", plugin="pyav")
        return out
    except Exception as e:  # noqa: BLE001
        gif = out.with_suffix(".gif")
        print(f"  (mp4 failed: {type(e).__name__}: {e} -> writing {gif.name})")
        iio.imwrite(gif, frames, duration=1.0 / fps, loop=0)
        return gif


def render_clip(clip_id: str, out_dir: Path, fps: float) -> Path:
    src = KIMODO_REP / f"{clip_id}.npz"
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    kd = _load_kimodo_npz(src)
    ud = kimodo_to_uniego(kd)
    kd2 = uniego_to_kimodo(ud)

    gt = kd["posed_joints"].cpu().numpy()        # (T, 22, 3)
    gen = kd2["posed_joints"].cpu().numpy()      # (T, 22, 3)
    maxd = float(np.abs(gt - gen).max())
    meand = float(np.abs(gt - gen).mean())

    text = _caption_for(clip_id)
    caption = f"{clip_id}: {text} | uniego round-trip max|Δ|={maxd:.1e}m mean={meand:.1e}m"
    frames = render_sidebyside(gt, gen, caption=caption)
    out = _save(frames, out_dir / f"{clip_id}_uniego_roundtrip.mp4", fps)
    print(f"  {clip_id}: max|Δpos|={maxd:.2e}m -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="*", default=None, help="Clip ids (e.g. 000004 M000123).")
    ap.add_argument("--n", type=int, default=None, help="Instead of --ids, sample N random clips.")
    ap.add_argument("--out-dir", type=Path, default=_THIS_DIR / "uniego_viz")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.n is not None:
        import random

        random.seed(args.seed)
        files = sorted(KIMODO_REP.glob("*.npz"))
        ids = [random.choice(files).stem for _ in range(args.n)]
    else:
        ids = args.ids or DEFAULT_IDS

    print(f"Rendering {len(ids)} clip(s) to {args.out_dir}")
    for clip_id in ids:
        render_clip(clip_id, args.out_dir, args.fps)


if __name__ == "__main__":
    main()

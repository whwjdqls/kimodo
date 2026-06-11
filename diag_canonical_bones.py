"""Test fix: prime FK bone-length cache with canonical lengths from motion
012314 (long, well-behaved), then re-check FK vs positions on the outliers.
"""
import sys, numpy as np, torch
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

sys.path.insert(0, '/home/jungbin_cho/kimodo_open')
sys.path.insert(0, '/home/jungbin_cho/kimodo_open/benchmark')
sys.path.insert(0, '/home/jungbin_cho/HumanML3D')

from omegaconf import OmegaConf
from kimodo.scripts.train import KimodoLoss, build_denoiser_from_model_config
from kimodo.motion_rep.fk_hml3d import (
    derive_bone_lengths_from_world_joints,
    world_joints_from_kimodo_features,
)

cfg = OmegaConf.load('/home/jungbin_cho/kimodo_open/runs/mdm_hml3d_kimrep_fp32/config.yaml')
OmegaConf.resolve(cfg)
denoiser = build_denoiser_from_model_config(
    cfg.model_config_path, cfg.stats_path, fps_override=cfg.get('denoiser_fps_override'),
).eval()
fk = KimodoLoss(motion_rep=denoiser.motion_rep, weights={}, fk_kind='chainreset_hml3d').eval()
mr = denoiser.motion_rep

# Derive canonical bone lengths from the long, canonical motion 012314.
canon_kim = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/012314.npz')
canon_features = torch.from_numpy(canon_kim['features']).float()
canon_world = world_joints_from_kimodo_features(canon_features, mr.slice_dict, n_joints=22)
canonical_bone_lengths = derive_bone_lengths_from_world_joints(canon_world.unsqueeze(0)).squeeze(0)
print(f'Canonical bone lengths (from motion 012314, T={canon_features.shape[0]}):')
print(f'  shape={canonical_bone_lengths.shape}, dtype={canonical_bone_lengths.dtype}')
print(f'  first 5 joints: {canonical_bone_lengths[:5].numpy()}')

print()
print(f'{"motion":12s}  {"per-motion FK err":>20s}  {"canonical-cache FK err":>22s}')
for mid in ['S_M004498', '003574', 'M006545', 'M006904']:
    kim = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{mid}.npz')
    features = torch.from_numpy(kim['features']).float()
    pos_block_joints = world_joints_from_kimodo_features(features, mr.slice_dict, n_joints=22)

    # Per-motion derived (current behavior — cold path)
    fk._fk_bone_lengths_cache = None
    fk_per_motion = fk._fk_world_from_pred(features.unsqueeze(0))[0]
    err_per = float((fk_per_motion - pos_block_joints).abs().max())

    # Canonical cached
    fk._fk_bone_lengths_cache = canonical_bone_lengths.clone()
    fk_canon = fk._fk_world_from_pred(features.unsqueeze(0))[0]
    err_canon = float((fk_canon - pos_block_joints).abs().max())

    print(f'{mid:12s}  {err_per:>20.3e}  {err_canon:>22.3e}')

# Reset
fk._fk_bone_lengths_cache = None

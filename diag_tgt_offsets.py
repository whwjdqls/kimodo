"""Test the tgt_offsets hypothesis:

1. Derive tgt_offsets (parent-to-child bone VECTORS) from motion 012314 frame 0,
   the same way HML3D's encoder did.
2. Derive raw_offsets * bone_lengths (kimodo's chain-reset FK convention).
3. Compare them per joint — if they differ, that's our source of FK error.
4. Patch the FK to use tgt_offsets directly, re-run outliers, see if fixed.
"""
import sys, numpy as np, torch
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

sys.path.insert(0, '/home/jungbin_cho/kimodo_open')
sys.path.insert(0, '/home/jungbin_cho/kimodo_open/benchmark')
sys.path.insert(0, '/home/jungbin_cho/HumanML3D')

from omegaconf import OmegaConf
from common.skeleton import Skeleton
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain

from kimodo.scripts.train import KimodoLoss, build_denoiser_from_model_config
from kimodo.motion_rep.fk_hml3d import (
    derive_bone_lengths_from_world_joints,
    world_joints_from_kimodo_features,
    HML3D_RAW_OFFSETS,
    HML3D_KINEMATIC_CHAIN,
)


# ---------- 1) Derive tgt_offsets from motion 012314 ----------
canon_joints = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/new_joints/012314.npy')   # (T, 22, 3)
skel = Skeleton(torch.from_numpy(t2m_raw_offsets).float(), t2m_kinematic_chain, 'cpu')
tgt_offsets = skel.get_offsets_joints(torch.from_numpy(canon_joints[0]).float())  # (22, 3) bone vectors
print(f'tgt_offsets shape: {tgt_offsets.shape}')

# ---------- 2) Derive raw_offsets * bone_lengths ----------
cfg = OmegaConf.load('/home/jungbin_cho/kimodo_open/runs/mdm_hml3d_kimrep_fp32/config.yaml')
OmegaConf.resolve(cfg)
denoiser = build_denoiser_from_model_config(
    cfg.model_config_path, cfg.stats_path, fps_override=cfg.get('denoiser_fps_override'),
).eval()
mr = denoiser.motion_rep
canon_kim = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/012314.npz')
canon_world = world_joints_from_kimodo_features(
    torch.from_numpy(canon_kim['features']).float(), mr.slice_dict, n_joints=22,
)
canon_bl = derive_bone_lengths_from_world_joints(canon_world.unsqueeze(0)).squeeze(0)  # (22,)
raw_x_len = HML3D_RAW_OFFSETS * canon_bl[:, None]  # (22, 3)

# ---------- 3) Compare per joint ----------
print(f'\n{"joint":5s}  {"tgt_offset (HML3D)":>30s}  {"raw*length (kimodo)":>30s}  {"diff":>10s}')
for j in range(22):
    t = tgt_offsets[j].numpy()
    r = raw_x_len[j].numpy()
    diff = np.linalg.norm(t - r)
    marker = '   <-- differs' if diff > 1e-4 else ''
    print(f'  {j:3d}  [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}]  [{r[0]:+.4f} {r[1]:+.4f} {r[2]:+.4f}]  {diff:10.4f}{marker}')

# ---------- 4) Patch FK to use tgt_offsets, re-test outliers ----------
print('\n--- patched FK test ---')

def fk_with_tgt_offsets(global_rot_mats, root_pos, offsets, chains):
    """Re-implement chainreset_fk_world_joints using bone VECTORS (offsets)
    directly, instead of raw_offsets * scalar bone_lengths."""
    if global_rot_mats.dim() == 4:
        global_rot_mats = global_rot_mats.unsqueeze(0)
        root_pos = root_pos.unsqueeze(0)
    B, T, J, _, _ = global_rot_mats.shape
    offsets = offsets.to(device=global_rot_mats.device, dtype=global_rot_mats.dtype)  # (J, 3)
    world_pos = torch.zeros(B, T, J, 3, device=global_rot_mats.device, dtype=global_rot_mats.dtype)
    world_pos[:, :, 0] = root_pos
    for chain in chains:
        if len(chain) < 2:
            continue
        for k in range(1, len(chain)):
            parent_idx = int(chain[k - 1])
            child_idx = int(chain[k])
            rotated = torch.einsum('btij,j->bti', global_rot_mats[:, :, child_idx], offsets[child_idx])
            world_pos[:, :, child_idx] = world_pos[:, :, parent_idx] + rotated
    return world_pos.squeeze(0)

# Manually compute global_rot_mats from kimodo features, then run patched FK.
from kimodo.geometry import cont6d_to_matrix
fk_helper = KimodoLoss(motion_rep=mr, weights={}, fk_kind='chainreset_hml3d').eval()

for mid in ['S_M004498', '003574', 'M006545', 'M006904']:
    kim = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{mid}.npz')
    features = torch.from_numpy(kim['features']).float()
    pos_block_joints = world_joints_from_kimodo_features(features, mr.slice_dict, n_joints=22)

    # Build global_rot_mats from features
    rot6d = features[:, mr.slice_dict['global_rot_data']].reshape(-1, 22, 6)
    grm = cont6d_to_matrix(rot6d)  # (T, 22, 3, 3)

    # actual_root from features
    smooth_root = features[:, mr.slice_dict['smooth_root_pos']]
    local_jp = features[:, mr.slice_dict['local_joints_positions']].reshape(-1, 22, 3)
    root_idx = 0
    actual_root = torch.stack([
        smooth_root[:, 0] + local_jp[:, root_idx, 0],
        local_jp[:, root_idx, 1],
        smooth_root[:, 2] + local_jp[:, root_idx, 2],
    ], dim=-1)  # (T, 3)

    # Original FK (raw_offsets * scalar)
    fk_orig = fk_helper._fk_world_from_pred(features.unsqueeze(0))[0]

    # Patched FK (tgt_offsets vectors)
    fk_patched = fk_with_tgt_offsets(grm, actual_root, tgt_offsets, HML3D_KINEMATIC_CHAIN)

    err_orig = float((fk_orig - pos_block_joints).abs().max())
    err_patched = float((fk_patched - pos_block_joints).abs().max())
    print(f'  {mid:12s}  orig FK err={err_orig:.3e}  patched FK err={err_patched:.3e}')

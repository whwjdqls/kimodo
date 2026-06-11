"""Verify the heading-wrap fix in _delta_rot_vel_from_alpha.

Sweeps 30 random GT kimodo motions through our eval pipeline (kimodo features
-> FK -> kimodo_to_humanml3d) and reports per-block max errors vs the
original HumanML3D 263-D. The pre-fix run had 2/30 motions hitting
rot_velocity error = π and propagating through ric_data / local_velocity.
This run should show 0/30 such outliers.
"""
import os, sys, random
import numpy as np
import torch

sys.path.insert(0, '/home/jungbin_cho/kimodo_open')
sys.path.insert(0, '/home/jungbin_cho/kimodo_open/benchmark')
sys.path.insert(0, '/home/jungbin_cho/HumanML3D')

from omegaconf import OmegaConf
from kimodo.scripts.train import KimodoLoss, build_denoiser_from_model_config
from humanml3d_to_kimodo import kimodo_to_humanml3d
from evaluation.kimodo_decode import kimodo_features_to_decode_dict


cfg = OmegaConf.load('/home/jungbin_cho/kimodo_open/runs/mdm_hml3d_kimrep_fp32/config.yaml')
OmegaConf.resolve(cfg)
denoiser = build_denoiser_from_model_config(
    cfg.model_config_path, cfg.stats_path, fps_override=cfg.get('denoiser_fps_override'),
).eval()
fk_helper = KimodoLoss(
    motion_rep=denoiser.motion_rep, weights={},
    fk_kind='chainreset_hml3d', fk_target='gt',
).eval()
motion_rep = denoiser.motion_rep


def rebuild_eval_style(features_273):
    decode = kimodo_features_to_decode_dict(
        features_273, motion_rep.slice_dict, n_joints=motion_rep.nbjoints,
    )
    fk_joints = fk_helper._fk_world_from_pred(features_273.unsqueeze(0))[0]
    decode['posed_joints'] = fk_joints
    decode['root_positions'] = fk_joints[:, 0]
    fk_vel = torch.zeros_like(fk_joints)
    fk_vel[:-1] = fk_joints[1:] - fk_joints[:-1]
    decode['velocities'] = fk_vel
    return kimodo_to_humanml3d(decode, device='cpu'), fk_joints


ids = sorted(
    f[:-4] for f in os.listdir('/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep')
    if f.endswith('.npz')
)
random.seed(42)
test_ids = random.sample(ids, 30)

blocks = {
    'rot_velocity': (0, 1), 'lin_velocity': (1, 3), 'root_height': (3, 4),
    'ric_data': (4, 67), 'rot_data': (67, 193),
    'local_velocity': (193, 259), 'foot_contacts': (259, 263),
}
per_block_maxes = {k: [] for k in blocks}
heading_wrap_motions = []
ric_outliers = []

for mid in test_ids:
    try:
        kim = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{mid}.npz')
        hml_orig = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{mid}.npy')
    except FileNotFoundError:
        continue
    features_273 = torch.from_numpy(kim['features']).float()
    hml_rebuilt, _ = rebuild_eval_style(features_273)
    T_min = min(hml_orig.shape[0], hml_rebuilt.shape[0])
    diff = np.abs(hml_orig[:T_min - 1] - hml_rebuilt.cpu().numpy()[:T_min - 1])
    for name, (lo, hi) in blocks.items():
        per_block_maxes[name].append(float(diff[..., lo:hi].max()))
    if float(diff[..., 0:1].max()) > 1.0:
        heading_wrap_motions.append((mid, float(diff[..., 0:1].max())))
    if float(diff[..., 4:67].max()) > 1e-3:
        ric_outliers.append((mid, float(diff[..., 4:67].max())))


print(f'\nTested {len(per_block_maxes["ric_data"])} motions\n')
print(f'{"block":18s}  {"median":>10s}  {"p90":>10s}  {"max":>10s}')
for name, vals in per_block_maxes.items():
    a = np.array(vals)
    print(f'{name:18s}  {np.median(a):10.3e}  {np.percentile(a, 90):10.3e}  {a.max():10.3e}')

print()
print(f'heading-wrap motions (rot_velocity err >1.0, pre-fix had ≈π = 3.142): {len(heading_wrap_motions)}/{len(test_ids)}')
for mid, mx in heading_wrap_motions[:5]:
    print(f'  {mid}: rot_vel max err = {mx:.3e}')

print()
print(f'ric_data outliers (>1e-3): {len(ric_outliers)}/{len(test_ids)}')
for mid, mx in ric_outliers[:5]:
    print(f'  {mid}: ric max err = {mx:.3e}')

print()
if not heading_wrap_motions and len(ric_outliers) <= 1:
    print('PASS: heading-wrap is fixed and ric_data outliers (if any) are isolated.')
elif not heading_wrap_motions:
    print('PARTIAL: heading-wrap is fixed; some ric outliers remain (separate cause).')
else:
    print('FAIL: heading-wrap still present.')

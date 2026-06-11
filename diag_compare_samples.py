"""Confirm we're comparing the SAME 30 motions before and after the fix.

The pre-fix test used `ids = [...] (unsorted)`, the post-fix test used
`ids = sorted(...)`. random.sample(seed=42) gives different motions for
each ordering, so the two outlier sets aren't directly comparable.

Re-run with both orderings to see the failing motions in each.
"""
import sys, os, random, numpy as np, torch
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

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
fk_helper = KimodoLoss(motion_rep=denoiser.motion_rep, weights={}, fk_kind='chainreset_hml3d').eval()
motion_rep = denoiser.motion_rep

def rebuild(features_273):
    decode = kimodo_features_to_decode_dict(features_273, motion_rep.slice_dict, n_joints=22)
    fk_joints = fk_helper._fk_world_from_pred(features_273.unsqueeze(0))[0]
    decode['posed_joints'] = fk_joints
    decode['root_positions'] = fk_joints[:, 0]
    fk_vel = torch.zeros_like(fk_joints)
    fk_vel[:-1] = fk_joints[1:] - fk_joints[:-1]
    decode['velocities'] = fk_vel
    return kimodo_to_humanml3d(decode, device='cpu')

# Use BOTH orderings to get both sample sets.
unsorted_ids = [f[:-4] for f in os.listdir('/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep') if f.endswith('.npz')]
sorted_ids = sorted(unsorted_ids)

for label, ids in [('UNSORTED (matches pre-fix sample)', unsorted_ids),
                   ('SORTED (matches post-fix sample)', sorted_ids)]:
    print(f'\n=== {label} ===')
    random.seed(42)
    test_ids = random.sample(ids, 30)
    outliers_rv = []
    outliers_ric = []
    for mid in test_ids:
        try:
            kim = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{mid}.npz')
            hml = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{mid}.npy')
        except FileNotFoundError:
            continue
        feats = torch.from_numpy(kim['features']).float()
        rebuilt = rebuild(feats).cpu().numpy()
        T_min = min(hml.shape[0], rebuilt.shape[0]) - 1
        diff = np.abs(hml[:T_min] - rebuilt[:T_min])
        rv_err = float(diff[..., 0:1].max())
        ric_err = float(diff[..., 4:67].max())
        if rv_err > 1.0:
            outliers_rv.append((mid, rv_err))
        if ric_err > 1e-3:
            outliers_ric.append((mid, ric_err))
    print(f'  rot_velocity outliers (>1.0):   {len(outliers_rv)}/30')
    for mid, e in outliers_rv: print(f'    {mid}: {e:.3f}')
    print(f'  ric_data outliers (>1e-3):       {len(outliers_ric)}/30')
    for mid, e in outliers_ric: print(f'    {mid}: {e:.3e}')

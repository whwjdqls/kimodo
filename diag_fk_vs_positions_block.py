"""Confirm: are FK joints != positions block for this outlier motion?"""
import sys, numpy as np, torch
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

sys.path.insert(0, '/home/jungbin_cho/kimodo_open')
sys.path.insert(0, '/home/jungbin_cho/kimodo_open/benchmark')
sys.path.insert(0, '/home/jungbin_cho/HumanML3D')

from omegaconf import OmegaConf
from kimodo.scripts.train import KimodoLoss, build_denoiser_from_model_config
from humanml3d_to_kimodo import recover_from_ric
from kimodo.motion_rep.fk_hml3d import world_joints_from_kimodo_features

cfg = OmegaConf.load('/home/jungbin_cho/kimodo_open/runs/mdm_hml3d_kimrep_fp32/config.yaml')
OmegaConf.resolve(cfg)
denoiser = build_denoiser_from_model_config(
    cfg.model_config_path, cfg.stats_path, fps_override=cfg.get('denoiser_fps_override'),
).eval()
fk = KimodoLoss(motion_rep=denoiser.motion_rep, weights={}, fk_kind='chainreset_hml3d').eval()
mr = denoiser.motion_rep

for mid in ['S_M004498', '003574', 'M006545', 'M006904']:
    kim = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{mid}.npz')
    hml = torch.from_numpy(np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{mid}.npy')).float()
    features = torch.from_numpy(kim['features']).float()

    fk_joints = fk._fk_world_from_pred(features.unsqueeze(0))[0]                  # (T, 22, 3) via chain-reset FK
    pos_block_joints = world_joints_from_kimodo_features(features, mr.slice_dict, n_joints=22)  # (T, 22, 3) from positions block
    hml_joints = recover_from_ric(hml, joints_num=22)                              # (T, 22, 3) HML3D recovers

    # Drop last frame from kimodo (dummy) before comparing.
    T_min = min(fk_joints.shape[0], hml_joints.shape[0]) - 1

    d_fk_vs_pos = (fk_joints[:T_min] - pos_block_joints[:T_min]).abs()
    d_fk_vs_hml = (fk_joints[:T_min] - hml_joints[:T_min]).abs()
    d_pos_vs_hml = (pos_block_joints[:T_min] - hml_joints[:T_min]).abs()
    print(f'\n{mid} (T={features.shape[0]}):')
    print(f'  FK joints vs positions block:          max={d_fk_vs_pos.max():.3e}  mean={d_fk_vs_pos.mean():.3e}')
    print(f'  FK joints vs HML3D recover_from_ric:   max={d_fk_vs_hml.max():.3e}  mean={d_fk_vs_hml.mean():.3e}')
    print(f'  positions block vs HML3D recover:      max={d_pos_vs_hml.max():.3e}  mean={d_pos_vs_hml.mean():.3e}')
    # Which joint is worst in FK?
    worst_per_joint = d_fk_vs_hml.reshape(T_min, 22, 3).norm(dim=-1).max(dim=0).values  # (22,)
    j_worst = int(worst_per_joint.argmax())
    print(f'  worst joint in FK: j={j_worst}, worst pos err={float(worst_per_joint.max()):.3e} m')

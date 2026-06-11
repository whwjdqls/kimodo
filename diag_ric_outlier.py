"""Pin down where the ric_data outlier comes from for a single failing motion.

Check, in order, each link in the chain:
  1. Are the kimodo features' global_rot_mats[:, 0] actually pure-Y rotation
     matrices? (If not, atan2 extraction loses X/Z components.)
  2. Does our extracted alpha match the heading derived directly from the
     hml_orig 263's stored cumulative rot_velocity?
  3. Does our recovered r_rot_quat match the hml_orig's recovered r_rot_quat?
  4. Does our recovered world joints (root only) match hml_orig's?
  5. At what frame does ric_data first diverge?
"""
import sys, numpy as np, torch, math
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

sys.path.insert(0, '/home/jungbin_cho/kimodo_open')
sys.path.insert(0, '/home/jungbin_cho/kimodo_open/benchmark')
sys.path.insert(0, '/home/jungbin_cho/HumanML3D')

from omegaconf import OmegaConf
from kimodo.scripts.train import KimodoLoss, build_denoiser_from_model_config
from humanml3d_to_kimodo import (
    kimodo_to_humanml3d, recover_root_rot_pos, recover_from_ric,
    _matrix_to_quaternion_y_aligned, _delta_rot_vel_from_alpha,
)
from evaluation.kimodo_decode import kimodo_features_to_decode_dict


cfg = OmegaConf.load('/home/jungbin_cho/kimodo_open/runs/mdm_hml3d_kimrep_fp32/config.yaml')
OmegaConf.resolve(cfg)
denoiser = build_denoiser_from_model_config(
    cfg.model_config_path, cfg.stats_path, fps_override=cfg.get('denoiser_fps_override'),
).eval()
fk = KimodoLoss(motion_rep=denoiser.motion_rep, weights={}, fk_kind='chainreset_hml3d').eval()
mr = denoiser.motion_rep

mid = 'S_M004498'   # the worst outlier (1.14 m)
kim = np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/kimodo_rep/{mid}.npz')
hml_orig = torch.from_numpy(np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{mid}.npy')).float()
features_273 = torch.from_numpy(kim['features']).float()
T = features_273.shape[0]
print(f'Motion {mid}, T={T} frames')

# 1) Is global_rot_mats[:, 0] pure-Y?
decode = kimodo_features_to_decode_dict(features_273, mr.slice_dict, n_joints=22)
grm0 = decode['global_rot_mats'][:, 0]   # (T, 3, 3)
# Pure-Y means rows 1 = [0,1,0], col 1 = [0,1,0] (the Y axis is invariant)
non_y = max(
    float(grm0[..., 1, 0].abs().max()), float(grm0[..., 1, 2].abs().max()),
    float(grm0[..., 0, 1].abs().max()), float(grm0[..., 2, 1].abs().max()),
    float((grm0[..., 1, 1] - 1.0).abs().max()),
)
print(f'1) global_rot_mats[:, 0] off-Y-axis components: max={non_y:.3e}  ' + ('PURE-Y ✓' if non_y < 1e-4 else 'NOT PURE-Y ✗'))

# 2) Compare our extracted alpha vs hml_orig's stored cumulative rot_velocity
r_rot_quat_ours = _matrix_to_quaternion_y_aligned(grm0)
alpha_ours = torch.atan2(r_rot_quat_ours[..., 2], r_rot_quat_ours[..., 0])
r_rot_quat_orig, r_pos_orig = recover_root_rot_pos(hml_orig)
alpha_orig = torch.atan2(r_rot_quat_orig[..., 2], r_rot_quat_orig[..., 0])
alpha_diff = (alpha_ours - alpha_orig).abs()
print(f'2) alpha_ours vs alpha_orig: max={alpha_diff.max():.3e}, mean={alpha_diff.mean():.3e}, alpha_diff[0]={float(alpha_ours[0]-alpha_orig[0]):.3e}')

# 3) Compare r_rot_quat values (both as matrices to be sign-invariant)
def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],     dim=-1),
        torch.stack([2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],     dim=-1),
        torch.stack([2*(x*z-w*y),   2*(y*z+w*x),     1-2*(x*x+y*y)],   dim=-1),
    ], dim=-2)
    return R
R_ours = quat_to_mat(r_rot_quat_ours)
R_orig = quat_to_mat(r_rot_quat_orig)
R_diff = (R_ours - R_orig).abs()
print(f'3) r_rot_quat (as matrix) diff: max={R_diff.max():.3e}, mean={R_diff.mean():.3e}')

# 4) Compare FK joints' root vs hml_orig's r_pos
fk_joints = fk._fk_world_from_pred(features_273.unsqueeze(0))[0]
root_diff = (fk_joints[:, 0] - r_pos_orig).abs()
print(f'4) FK joints[:, 0] vs hml_orig r_pos: max={root_diff.max():.3e}, mean={root_diff.mean():.3e}')

# 5) Where does ric_data diverge? Rebuild and find the bad frames.
decode['posed_joints'] = fk_joints
decode['root_positions'] = fk_joints[:, 0]
fk_vel = torch.zeros_like(fk_joints)
fk_vel[:-1] = fk_joints[1:] - fk_joints[:-1]
decode['velocities'] = fk_vel
hml_rebuilt = kimodo_to_humanml3d(decode, device='cpu')
T_min = min(hml_orig.shape[0], hml_rebuilt.shape[0]) - 1
diff = (hml_orig[:T_min] - hml_rebuilt[:T_min]).abs()
ric_diff = diff[..., 4:67]                            # (T-1, 63)
ric_per_frame = ric_diff.max(dim=-1).values            # (T-1,)
top5 = torch.topk(ric_per_frame, 5)
print(f'5) ric_data per-frame max diff — top 5 frames:')
for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
    print(f'   frame {int(idx):4d}: max ric diff = {float(val):.3e}  '
          f'(alpha_orig={float(alpha_orig[idx]):.4f}, alpha_ours={float(alpha_ours[idx]):.4f}, '
          f'rotvel_orig={float(hml_orig[idx, 0]):.4f}, rotvel_ours={float(hml_rebuilt[idx, 0]):.4f})')

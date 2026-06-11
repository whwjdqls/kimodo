"""Test whether HML3D's OWN stored rot_data is self-consistent with its
stored ric_data for the outlier motions.

Compare:
  A. recover_from_ric(263)   — positions directly from ric_data block
  B. recover_from_rot(263)   — positions from FK on rot_data block via HML3D's own
                                forward_kinematics_cont6d (NOT kimodo's FK)

If A == B: HML3D's data is self-consistent → bug is in kimodo's chain-reset FK
If A != B: HML3D's data itself is inconsistent → unfixable in our pipeline
"""
import sys
import numpy as np
import torch
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'int'): np.int = int

sys.path.insert(0, '/home/jungbin_cho/HumanML3D')
sys.path.insert(0, '/home/jungbin_cho/kimodo_open/benchmark')

from common.skeleton import Skeleton
from common.quaternion import quaternion_to_cont6d
from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from humanml3d_to_kimodo import recover_from_ric, recover_root_rot_pos


# Build HML3D's Skeleton with offsets from the canonical example (motion 012314).
canon_joints = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/new_joints/012314.npy')
skel = Skeleton(torch.from_numpy(t2m_raw_offsets).float(), t2m_kinematic_chain, 'cpu')
tgt_offsets = skel.get_offsets_joints(torch.from_numpy(canon_joints[0]).float())
skel.set_offset(tgt_offsets)


def recover_from_rot_hml3d(data):
    """HML3D's exact recover_from_rot logic — uses skel.forward_kinematics_cont6d."""
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)            # (T, 6)
    cont6d_params = data[..., 1 + 2 + 1 + 21 * 3 : 1 + 2 + 1 + 21 * 3 + 21 * 6]  # (T, 126)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)  # (T, 132)
    cont6d_params = cont6d_params.view(-1, 22, 6)
    positions = skel.forward_kinematics_cont6d(cont6d_params, r_pos)
    return positions


for mid in ['S_M004498', '003574', 'M006545', 'M006904']:
    hml = torch.from_numpy(np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{mid}.npy')).float()
    a = recover_from_ric(hml, joints_num=22)        # A: from positions block
    b = recover_from_rot_hml3d(hml)                  # B: from rotations via HML3D's own FK
    d_ab = (a - b).abs().max()
    print(f'  {mid}: HML3D recover_from_ric vs HML3D recover_from_rot  max|Δ| = {float(d_ab):.3e}')

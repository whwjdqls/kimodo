"""Run HML3D's recover_from_rot vs recover_from_ric over a large random
sample of HumanML3D test motions to characterize the discrepancy rate.

This tests ONLY HumanML3D's own data — no kimodo involvement at all.
If the discrepancy hits N% of motions, that's the upstream bug.
"""
import sys, os, random
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


canon = np.load('/home/jungbin_cho/HumanML3D/HumanML3D/new_joints/012314.npy')
skel = Skeleton(torch.from_numpy(t2m_raw_offsets).float(), t2m_kinematic_chain, 'cpu')
skel.set_offset(skel.get_offsets_joints(torch.from_numpy(canon[0]).float()))


def recover_from_rot_hml3d(data):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    r_rot_6d = quaternion_to_cont6d(r_rot_quat)
    rot_data = data[..., 1 + 2 + 1 + 21 * 3 : 1 + 2 + 1 + 21 * 3 + 21 * 6]
    cont6d = torch.cat([r_rot_6d, rot_data], dim=-1).view(-1, 22, 6)
    return skel.forward_kinematics_cont6d(cont6d, r_pos)


ids = sorted(f[:-4] for f in os.listdir('/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs') if f.endswith('.npy'))
print(f'Total motions in new_joint_vecs/: {len(ids)}')

# Use HML3D's official test split for a representative sample.
with open('/home/jungbin_cho/HumanML3D/HumanML3D/test.txt') as f:
    test_ids = [ln.strip() for ln in f if ln.strip()]
# Filter to those that actually exist on disk
test_ids = [m for m in test_ids if os.path.exists(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{m}.npy')]
print(f'Test split motions on disk: {len(test_ids)}')

N = min(500, len(test_ids))
random.seed(0)
sample = random.sample(test_ids, N)

errs = []
for i, mid in enumerate(sample):
    try:
        hml = torch.from_numpy(np.load(f'/home/jungbin_cho/HumanML3D/HumanML3D/new_joint_vecs/{mid}.npy')).float()
        a = recover_from_ric(hml, joints_num=22)
        b = recover_from_rot_hml3d(hml)
        d = float((a - b).abs().max())
        errs.append((mid, d))
    except Exception as e:
        print(f'  {mid}: skipped ({type(e).__name__})')

errs.sort(key=lambda x: x[1], reverse=True)
err_vals = np.array([e for _, e in errs])

print(f'\nTested {len(errs)} test motions')
print(f'Distribution of max |recover_from_ric − recover_from_rot| per motion:')
print(f'  min     : {err_vals.min():.3e}')
print(f'  median  : {np.median(err_vals):.3e}')
print(f'  p75     : {np.percentile(err_vals, 75):.3e}')
print(f'  p90     : {np.percentile(err_vals, 90):.3e}')
print(f'  p95     : {np.percentile(err_vals, 95):.3e}')
print(f'  p99     : {np.percentile(err_vals, 99):.3e}')
print(f'  max     : {err_vals.max():.3e}')
print()
for thr in (1e-3, 1e-2, 1e-1, 0.5, 1.0):
    n = int((err_vals > thr).sum())
    print(f'  motions with err > {thr:>6}: {n:4d} / {len(err_vals)}  ({100*n/len(err_vals):.1f}%)')

print(f'\nTop 10 worst:')
for mid, e in errs[:10]:
    print(f'  {mid}: {e:.3e}')

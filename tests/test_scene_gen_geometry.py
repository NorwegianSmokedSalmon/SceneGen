"""Geometry regression checks for SAM 3D placement (run in scene_gen_3d)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/segmentation'))
from generate_instance_views import get_lookat_matrix_opencv, calculate_tight_distance
from generate_sam3d_objects import world_splats, fit_instance_bounds


class SceneGeometryTests(unittest.TestCase):
    def test_cameras_are_proper_rotations_and_face_target(self):
        for position in ([2, 3, 4], [0, 0, 1], [0, 0, -1], [2, 0, 0]):
            c2w = get_lookat_matrix_opencv(np.array(position), np.zeros(3))
            np.testing.assert_allclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-5)
            np.testing.assert_allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5)
            target = np.linalg.inv(c2w) @ [0, 0, 0, 1]
            np.testing.assert_allclose(target[:2], 0, atol=1e-5)
            self.assertGreater(target[2], 0)

    def test_off_axis_camera_fits_all_box_corners(self):
        corners = np.array([[x,y,z] for x in [-1,1] for y in [-2,2] for z in [-3,3]], dtype=np.float32)
        direction = np.array([1., 2., 3.], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        fov = np.deg2rad(70)
        distance = calculate_tight_distance(corners, direction, fov)
        c2w = get_lookat_matrix_opencv(direction * distance, np.zeros(3))
        points = (corners - c2w[:3, 3]) @ c2w[:3, :3]
        self.assertTrue(np.all(points[:, 2] > 0))
        self.assertLessEqual(float(np.abs(points[:, :2] / points[:, 2:]).max()), np.tan(fov/2) + 1e-4)

    def test_known_pose_and_anisotropic_covariance(self):
        gaussian = SimpleNamespace(
            get_xyz=torch.tensor([[1., 0, 0]]), get_scaling=torch.tensor([[1., 2, 3]]),
            get_rotation=torch.tensor([[1., 0, 0, 0]]), get_features=torch.ones(1,1,3),
            get_opacity=torch.tensor([[.5]]))
        result = dict(gaussian=[gaussian], scale=torch.tensor([[2.,2,2]]),
                      rotation=torch.tensor([[2**-.5,0,0,2**-.5]]),
                      translation=torch.tensor([[0.,0,5.]]))
        c2w = np.eye(4); c2w[:3,3] = [10,20,30]
        out = world_splats(result, {'c2w':c2w.tolist()}, 15)
        torch.testing.assert_close(out['means'], torch.tensor([[10.,22,35]]))
        rotation = quaternion_to_matrix(out['quats'])[0]
        covariance = rotation @ torch.diag(out['scales'][0].exp().square()) @ rotation.T
        torch.testing.assert_close(covariance, torch.diag(torch.tensor([16.,4,36])), atol=1e-4, rtol=1e-5)

    def test_instance_anchor_corrects_scale_translation_and_kernel_size(self):
        points = torch.tensor([[x,y,z] for x in [-1.,1.] for y in [-2.,2.] for z in [-3.,3.]]).repeat(2,1)
        target = points * 2 + torch.tensor([10.,20,30])
        splats = {'means':points, 'scales':torch.zeros_like(points), 'opacities':torch.ones(len(points))}
        fitted, report = fit_instance_bounds(splats, {'means':target}, np.ones(len(points),dtype=int), 1)
        torch.testing.assert_close(fitted['means'], target)
        torch.testing.assert_close(fitted['scales'].exp(), torch.full_like(points,2))
        self.assertAlmostEqual(report['scale'], 2)
        torch.testing.assert_close(splats['means'], points)


if __name__ == '__main__':
    unittest.main()

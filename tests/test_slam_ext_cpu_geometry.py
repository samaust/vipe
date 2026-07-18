import unittest

import torch

from vipe.ext import slam_ext


def _inputs(device: str):
    poses = torch.zeros(4, 7, dtype=torch.float32, device=device)
    poses[:, 6] = 1.0
    poses[:, 0] = torch.linspace(0.0, 0.03, 4, device=device)
    disps = torch.full((4, 4, 5), 0.8, dtype=torch.float32, device=device)
    intrinsics = torch.tensor([20.0, 21.0, 2.0, 1.5], dtype=torch.float32, device=device)
    return poses.contiguous(), disps.contiguous(), intrinsics.contiguous()


class SlamExtCPUGeometryTest(unittest.TestCase):
    def test_identity_iproj_and_projmap(self):
        poses, disps, intrinsics = _inputs("cpu")
        poses[:, :3].zero_()
        disps.fill_(1.0)

        points = slam_ext.iproj(poses, disps, intrinsics)
        self.assertEqual(tuple(points.shape), (4, 4, 5, 3))
        torch.testing.assert_close(points[0, 0, 0], torch.tensor([-0.1, -1.5 / 21.0, 1.0]))

        ii = torch.tensor([0, 2], dtype=torch.long)
        jj = torch.tensor([0, 2], dtype=torch.long)
        coords, valid = slam_ext.projmap(poses, disps, intrinsics, ii, jj)
        expected_y, expected_x = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
        torch.testing.assert_close(coords[:, :, :, 0], expected_x.float().expand(2, -1, -1))
        torch.testing.assert_close(coords[:, :, :, 1], expected_y.float().expand(2, -1, -1))
        self.assertTrue(torch.all(valid == 1).item())

    def test_identity_frame_distance_is_zero(self):
        poses, disps, intrinsics = _inputs("cpu")
        poses[:, :3].zero_()
        camera_intrinsics = intrinsics.unsqueeze(0)
        indices = torch.tensor([0, 1, 2], dtype=torch.long)
        distance = slam_ext.frame_distance(
            poses,
            disps,
            camera_intrinsics,
            indices,
            indices,
            torch.zeros_like(indices),
            torch.zeros_like(indices),
            indices,
            0.3,
        )
        torch.testing.assert_close(distance, torch.zeros_like(distance), atol=1e-6, rtol=0.0)

    def test_depth_filter_runs_on_cpu(self):
        poses, disps, intrinsics = _inputs("cpu")
        inds = torch.arange(4, dtype=torch.long)
        count = slam_ext.depth_filter(poses, disps, intrinsics, inds, torch.full((4,), 0.1))
        self.assertEqual(tuple(count.shape), tuple(disps.shape))
        self.assertTrue(torch.isfinite(count).all().item())
        self.assertGreater(count.sum().item(), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU/CUDA parity")
    def test_matches_cuda(self):
        cpu = _inputs("cpu")
        gpu = tuple(value.cuda() for value in cpu)
        ii = torch.tensor([0, 2, 3], dtype=torch.long)
        jj = torch.tensor([1, 1, 0], dtype=torch.long)

        cpu_proj = slam_ext.projmap(*cpu, ii, jj)
        gpu_proj = slam_ext.projmap(*gpu, ii.cuda(), jj.cuda())
        for actual, expected in zip(cpu_proj, gpu_proj):
            torch.testing.assert_close(actual, expected.cpu(), atol=1e-5, rtol=1e-5)

        cpu_iproj = slam_ext.iproj(*cpu)
        gpu_iproj = slam_ext.iproj(*gpu)
        torch.testing.assert_close(cpu_iproj, gpu_iproj.cpu(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

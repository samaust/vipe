import unittest

import torch

from vipe.ext import droid_net_ext


class DroidExtCPUTest(unittest.TestCase):
    def test_corr_index_integer_coordinate(self):
        volume = torch.arange(3 * 4, dtype=torch.float32).reshape(1, 1, 1, 3, 4).contiguous()
        coords = torch.tensor([[[[2.0]], [[1.0]]]], dtype=torch.float32).contiguous()
        (corr,) = droid_net_ext.corr_index_forward(volume, coords, 1)
        # The native operator orders its radius axes as (dx, dy), matching CUDA.
        expected = volume[0, 0, 0, :, 1:4].T.contiguous()
        torch.testing.assert_close(corr[0, :, :, 0, 0], expected)

    def test_corr_index_backward(self):
        volume = torch.zeros(1, 1, 1, 3, 3)
        coords = torch.tensor([[[[1.5]], [[1.5]]]], dtype=torch.float32).contiguous()
        grad = torch.ones(1, 1, 1, 1, 1)
        (volume_grad,) = droid_net_ext.corr_index_backward(volume, coords, grad, 0)
        expected = torch.zeros_like(volume)
        expected[0, 0, 0, 1:3, 1:3] = 0.25
        torch.testing.assert_close(volume_grad, expected)

    def test_altcorr_indexed_matches_reference_kernel(self):
        torch.manual_seed(4)
        fmap = torch.randn(1, 2, 3, 4, 5)
        pyramid = [fmap, torch.nn.functional.avg_pool2d(fmap.permute(0, 1, 4, 2, 3).flatten(0, 1), 2).reshape(1, 2, 5, 1, 2).permute(0, 1, 3, 4, 2).contiguous()]
        coords = torch.rand(1, 2, 3, 4, 2)
        coords[..., 0] *= 3
        coords[..., 1] *= 2
        ii = torch.tensor([0, 1], dtype=torch.long)
        jj = torch.tensor([1, 0], dtype=torch.long)

        (indexed,) = droid_net_ext.altcorr_index_forward(fmap, pyramid, coords.contiguous(), ii, jj, 1)
        self.assertEqual(tuple(indexed.shape), (1, 2, 18, 3, 4))
        self.assertTrue(torch.isfinite(indexed).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU/CUDA parity")
    def test_matches_cuda(self):
        torch.manual_seed(9)
        volume = torch.randn(2, 3, 4, 5, 6)
        coords = torch.rand(2, 2, 3, 4)
        coords[:, 0] *= 5
        coords[:, 1] *= 4
        cpu_forward = droid_net_ext.corr_index_forward(volume.contiguous(), coords.contiguous(), 2)[0]
        gpu_forward = droid_net_ext.corr_index_forward(volume.cuda().contiguous(), coords.cuda().contiguous(), 2)[0]
        torch.testing.assert_close(cpu_forward, gpu_forward.cpu(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

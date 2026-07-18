import unittest

import torch

from vipe.ext import utils_ext


class UtilsExtCPUTest(unittest.TestCase):
    def test_nearest_neighbours_returns_squared_l2(self):
        tree = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=torch.float32)
        query = torch.tensor([[1.5, 0.0], [0.0, 2.0]], dtype=torch.float32)

        distance, index = utils_ext.nearest_neighbours(query, tree, 2)

        torch.testing.assert_close(distance, torch.tensor([[0.25, 2.25], [1.0, 4.0]]))
        torch.testing.assert_close(index, torch.tensor([[1, 0], [2, 0]], dtype=torch.int32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CPU/CUDA parity")
    def test_matches_cuda(self):
        torch.manual_seed(17)
        tree = torch.randn(32, 3)
        query = torch.randn(11, 3)
        cpu_distance, cpu_index = utils_ext.nearest_neighbours(query, tree, 4)
        gpu_distance, gpu_index = utils_ext.nearest_neighbours(query.cuda(), tree.cuda(), 4)
        torch.testing.assert_close(cpu_distance, gpu_distance.cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(cpu_index, gpu_index.cpu())


if __name__ == "__main__":
    unittest.main()

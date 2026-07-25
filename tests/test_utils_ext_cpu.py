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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the small-tree regression test")
    def test_cuda_supports_tree_smaller_than_leaf_capacity(self):
        tree = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        query = torch.tensor([[2.0, 2.0, 3.0]], device="cuda")

        distance, index = utils_ext.nearest_neighbours(query, tree, 1)

        torch.testing.assert_close(distance.cpu(), torch.tensor([[1.0]]))
        torch.testing.assert_close(index.cpu(), torch.tensor([[0]], dtype=torch.int32))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CUDA input validation")
    def test_cuda_rejects_empty_tree(self):
        with self.assertRaisesRegex(RuntimeError, "knn must be positive and no larger than the tree"):
            utils_ext.nearest_neighbours(torch.empty(1, 3, device="cuda"), torch.empty(0, 3, device="cuda"), 1)


if __name__ == "__main__":
    unittest.main()

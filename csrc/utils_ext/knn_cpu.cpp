/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> nearestNeighboursCpu(torch::Tensor query, torch::Tensor tree, int knn) {
    TORCH_CHECK(query.device().is_cpu(), "query must be a CPU tensor");
    TORCH_CHECK(tree.device().is_cpu(), "tree must be a CPU tensor");
    TORCH_CHECK(query.scalar_type() == torch::kFloat32, "query must be a float tensor");
    TORCH_CHECK(tree.scalar_type() == torch::kFloat32, "tree must be a float tensor");
    TORCH_CHECK(query.dim() == 2 && tree.dim() == 2, "query and tree must be rank-2 tensors");
    TORCH_CHECK(query.size(1) == tree.size(1), "query and tree point dimensions must match");
    TORCH_CHECK(knn > 0 && tree.size(0) >= knn, "knn must be positive and no larger than the tree");

    // The CUDA implementation reports squared L2 distance, so avoid cdist's
    // square root and compute the same quantity directly.
    auto delta = query.unsqueeze(1) - tree.unsqueeze(0);
    auto squared_distance = delta.square().sum(-1);
    auto nearest = torch::topk(squared_distance, knn, -1, false, true);
    return {std::get<0>(nearest), std::get<1>(nearest).to(torch::kInt32)};
}

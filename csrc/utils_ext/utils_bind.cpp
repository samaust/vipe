/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

std::vector<torch::Tensor> nearestNeighboursCuda(torch::Tensor query, torch::Tensor tree, int knn);
std::vector<torch::Tensor> nearestNeighboursCpu(torch::Tensor query, torch::Tensor tree, int knn);

std::vector<torch::Tensor> nearestNeighbours(torch::Tensor query, torch::Tensor tree, int knn) {
    TORCH_CHECK(query.device() == tree.device(), "query and tree must be on the same device");
    return query.device().is_cpu() ? nearestNeighboursCpu(query, tree, knn)
                                   : nearestNeighboursCuda(query, tree, knn);
}

void pybind_utils_ext(py::module &m) {
    m.def("nearest_neighbours", &nearestNeighbours, "KNN computation; distance is squared L2.", py::arg("query"),
          py::arg("tree"), py::arg("knn"));
}

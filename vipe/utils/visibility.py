# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

from vipe.utils.depth import get_camera_rays


def invisible_pixels_from_depth(
    metric_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return pixels touching a perpendicular or back-facing depth-grid triangle."""
    if metric_depth.ndim != 2:
        raise ValueError(f"Metric depth must have shape (H, W), got {tuple(metric_depth.shape)}")
    if intrinsics.ndim != 1 or intrinsics.shape[0] != 4:
        raise ValueError(f"Pinhole intrinsics must have shape (4,), got {tuple(intrinsics.shape)}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Visibility threshold must be in [0, 1], got {threshold}")

    height, width = metric_depth.shape
    invisible = torch.zeros((height, width), dtype=torch.bool, device=metric_depth.device)
    if height < 2 or width < 2:
        return invisible

    depth = metric_depth.float()
    camera_intrinsics = intrinsics.to(device=depth.device, dtype=depth.dtype)
    if not torch.isfinite(camera_intrinsics).all() or (camera_intrinsics[:2] <= 0).any():
        raise ValueError("Pinhole intrinsics must be finite with positive focal lengths")

    xyz = get_camera_rays(height, width, camera_intrinsics) * depth.unsqueeze(-1)
    valid = torch.isfinite(depth) & (depth > 0)

    p00 = xyz[:-1, :-1]
    p10 = xyz[1:, :-1]
    p11 = xyz[1:, 1:]
    p01 = xyz[:-1, 1:]

    def qualifying_faces(
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        valid0: torch.Tensor,
        valid1: torch.Tensor,
        valid2: torch.Tensor,
    ) -> torch.Tensor:
        normals = torch.linalg.cross(p1 - p0, p2 - p0)
        normal_lengths = torch.linalg.vector_norm(normals, dim=-1)
        centroids = (p0 + p1 + p2) / 3.0
        centroid_lengths = torch.linalg.vector_norm(centroids, dim=-1)
        face_normals = F.normalize(normals, dim=-1)
        to_camera = F.normalize(-centroids, dim=-1)
        dot = torch.sum(face_normals * to_camera, dim=-1)
        epsilon = torch.finfo(depth.dtype).eps
        return (
            valid0
            & valid1
            & valid2
            & (normal_lengths > epsilon)
            & (centroid_lengths > epsilon)
            & torch.isfinite(dot)
            & (dot <= threshold)
        )

    first = qualifying_faces(
        p00,
        p10,
        p11,
        valid[:-1, :-1],
        valid[1:, :-1],
        valid[1:, 1:],
    )
    second = qualifying_faces(
        p00,
        p11,
        p01,
        valid[:-1, :-1],
        valid[1:, 1:],
        valid[:-1, 1:],
    )

    invisible[:-1, :-1] |= first
    invisible[1:, :-1] |= first
    invisible[1:, 1:] |= first
    invisible[:-1, :-1] |= second
    invisible[1:, 1:] |= second
    invisible[:-1, 1:] |= second
    return invisible

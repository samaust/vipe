/**
 * CPU implementations of the geometric DROID-SLAM operators.
 *
 * These intentionally mirror geom_kernels.cu.  Keeping the public signatures
 * identical lets slam.cpp dispatch solely from the input tensor device.
 */

#include <torch/extension.h>

#include <cmath>
#include <vector>

namespace slam_ext {
namespace {

constexpr float kMinDepth = 0.25f;

void check_cpu_float(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(value.scalar_type() == torch::kFloat32, name, " must have dtype float32");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void check_cpu_long(const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(value.scalar_type() == torch::kInt64, name, " must have dtype int64");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

inline void act_so3(const float* q, const float* x, float* y) {
    const float uv[3] = {
        2.0f * (q[1] * x[2] - q[2] * x[1]),
        2.0f * (q[2] * x[0] - q[0] * x[2]),
        2.0f * (q[0] * x[1] - q[1] * x[0]),
    };
    y[0] = x[0] + q[3] * uv[0] + (q[1] * uv[2] - q[2] * uv[1]);
    y[1] = x[1] + q[3] * uv[1] + (q[2] * uv[0] - q[0] * uv[2]);
    y[2] = x[2] + q[3] * uv[2] + (q[0] * uv[1] - q[1] * uv[0]);
}

inline void act_se3(const float* t, const float* q, const float* x, float* y) {
    act_so3(q, x, y);
    y[3] = x[3];
    y[0] += x[3] * t[0];
    y[1] += x[3] * t[1];
    y[2] += x[3] * t[2];
}

inline void relative_se3(const float* ti, const float* qi, const float* tj, const float* qj, float* tij,
                         float* qij) {
    qij[0] = -qj[3] * qi[0] + qj[0] * qi[3] - qj[1] * qi[2] + qj[2] * qi[1];
    qij[1] = -qj[3] * qi[1] + qj[1] * qi[3] - qj[2] * qi[0] + qj[0] * qi[2];
    qij[2] = -qj[3] * qi[2] + qj[2] * qi[3] - qj[0] * qi[1] + qj[1] * qi[0];
    qij[3] = qj[3] * qi[3] + qj[0] * qi[0] + qj[1] * qi[1] + qj[2] * qi[2];
    act_so3(qij, ti, tij);
    tij[0] = tj[0] - tij[0];
    tij[1] = tj[1] - tij[1];
    tij[2] = tj[2] - tij[2];
}

inline void load_relative_pose(const torch::TensorAccessor<float, 2>& poses, int64_t i, int64_t j, float* t,
                               float* q) {
    float ti[3], tj[3], qi[4], qj[4];
    for (int k = 0; k < 3; ++k) {
        ti[k] = poses[i][k];
        tj[k] = poses[j][k];
    }
    for (int k = 0; k < 4; ++k) {
        qi[k] = poses[i][k + 3];
        qj[k] = poses[j][k + 3];
    }
    relative_se3(ti, qi, tj, qj, t, q);
}

}  // namespace

torch::Tensor iproj_cpu(torch::Tensor poses, torch::Tensor disps, torch::Tensor intrinsics) {
    check_cpu_float(poses, "poses");
    check_cpu_float(disps, "disps");
    check_cpu_float(intrinsics, "intrinsics");
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape [N, 7]");
    TORCH_CHECK(disps.dim() == 3 && disps.size(0) == poses.size(0), "disps must have shape [N, H, W]");
    TORCH_CHECK(intrinsics.numel() >= 4, "intrinsics must contain fx, fy, cx, cy");

    const auto p = poses.accessor<float, 2>();
    const auto d = disps.accessor<float, 3>();
    const auto k = intrinsics.accessor<float, 1>();
    auto points = torch::zeros({disps.size(0), disps.size(1), disps.size(2), 3}, disps.options());
    auto out = points.accessor<float, 4>();
    const float fx = k[0], fy = k[1], cx = k[2], cy = k[3];

    for (int64_t n = 0; n < disps.size(0); ++n) {
        float t[3], q[4];
        for (int i = 0; i < 3; ++i) t[i] = p[n][i];
        for (int i = 0; i < 4; ++i) q[i] = p[n][i + 3];
        for (int64_t y = 0; y < disps.size(1); ++y) {
            for (int64_t x = 0; x < disps.size(2); ++x) {
                const float xi[4] = {(static_cast<float>(x) - cx) / fx, (static_cast<float>(y) - cy) / fy, 1.0f,
                                     d[n][y][x]};
                float xj[4];
                act_se3(t, q, xi, xj);
                out[n][y][x][0] = xj[0] / xj[3];
                out[n][y][x][1] = xj[1] / xj[3];
                out[n][y][x][2] = xj[2] / xj[3];
            }
        }
    }
    return points;
}

std::vector<torch::Tensor> projmap_cpu(torch::Tensor poses, torch::Tensor disps, torch::Tensor intrinsics,
                                       torch::Tensor ii, torch::Tensor jj) {
    check_cpu_float(poses, "poses");
    check_cpu_float(disps, "disps");
    check_cpu_float(intrinsics, "intrinsics");
    check_cpu_long(ii, "ii");
    check_cpu_long(jj, "jj");
    TORCH_CHECK(ii.numel() == jj.numel(), "ii and jj must have the same length");

    auto coords = torch::zeros({ii.size(0), disps.size(1), disps.size(2), 3}, poses.options());
    auto valid = torch::zeros({ii.size(0), disps.size(1), disps.size(2), 1}, poses.options());
    const auto p = poses.accessor<float, 2>();
    const auto d = disps.accessor<float, 3>();
    const auto k = intrinsics.accessor<float, 1>();
    const auto src = ii.accessor<int64_t, 1>();
    const auto dst = jj.accessor<int64_t, 1>();
    auto xy = coords.accessor<float, 4>();
    auto mask = valid.accessor<float, 4>();
    const float fx = k[0], fy = k[1], cx = k[2], cy = k[3];

    for (int64_t edge = 0; edge < ii.size(0); ++edge) {
        const int64_t i = src[edge], j = dst[edge];
        float t[3], q[4];
        load_relative_pose(p, i, j, t, q);
        for (int64_t y = 0; y < disps.size(1); ++y) {
            for (int64_t x = 0; x < disps.size(2); ++x) {
                const float xi[4] = {(static_cast<float>(x) - cx) / fx, (static_cast<float>(y) - cy) / fy, 1.0f,
                                     d[i][y][x]};
                float xj[4];
                act_se3(t, q, xi, xj);
                xy[edge][y][x][0] = static_cast<float>(x);
                xy[edge][y][x][1] = static_cast<float>(y);
                if (xj[2] > 0.01f) {
                    xy[edge][y][x][0] = fx * xj[0] / xj[2] + cx;
                    xy[edge][y][x][1] = fy * xj[1] / xj[2] + cy;
                }
                mask[edge][y][x][0] = xj[2] > kMinDepth ? 1.0f : 0.0f;
            }
        }
    }
    return {coords, valid};
}

torch::Tensor frame_distance_cpu(torch::Tensor poses, torch::Tensor disps, torch::Tensor intrinsics,
                                 torch::Tensor pi, torch::Tensor pj, torch::Tensor qi, torch::Tensor qj,
                                 torch::Tensor di, const float beta) {
    check_cpu_float(poses, "poses");
    check_cpu_float(disps, "disps");
    check_cpu_float(intrinsics, "intrinsics");
    check_cpu_long(pi, "pi");
    check_cpu_long(pj, "pj");
    check_cpu_long(qi, "qi");
    check_cpu_long(qj, "qj");
    check_cpu_long(di, "di");
    TORCH_CHECK(pi.numel() == pj.numel() && pi.numel() == qi.numel() && pi.numel() == qj.numel() &&
                    pi.numel() == di.numel(),
                "all frame_distance index tensors must have the same length");

    auto result = torch::zeros({pi.size(0)}, poses.options());
    const auto p = poses.accessor<float, 2>();
    const auto d = disps.accessor<float, 3>();
    const auto k = intrinsics.accessor<float, 2>();
    const auto src = pi.accessor<int64_t, 1>();
    const auto dst = pj.accessor<int64_t, 1>();
    const auto src_cam = qi.accessor<int64_t, 1>();
    const auto dst_cam = qj.accessor<int64_t, 1>();
    const auto disp_index = di.accessor<int64_t, 1>();
    auto dist = result.accessor<float, 1>();

    for (int64_t edge = 0; edge < pi.size(0); ++edge) {
        float t[3], q[4];
        load_relative_pose(p, src[edge], dst[edge], t, q);
        const auto si = src_cam[edge], sj = dst_cam[edge], sd = disp_index[edge];
        float accum = 0.0f, valid = 0.0f, total = 0.0f;
        for (int64_t y = 0; y < disps.size(1); ++y) {
            for (int64_t x = 0; x < disps.size(2); ++x) {
                const float u = static_cast<float>(x), v = static_cast<float>(y);
                const float xi[4] = {(u - k[si][2]) / k[si][0], (v - k[si][3]) / k[si][1], 1.0f, d[sd][y][x]};
                float xj[4];
                act_se3(t, q, xi, xj);
                float du = k[sj][0] * xj[0] / xj[2] + k[sj][2] - u;
                float dv = k[sj][1] * xj[1] / xj[2] + k[sj][3] - v;
                total += beta;
                if (xj[2] > kMinDepth) {
                    accum += beta * std::sqrt(du * du + dv * dv);
                    valid += beta;
                }
                xj[0] = xi[0] + xi[3] * t[0];
                xj[1] = xi[1] + xi[3] * t[1];
                xj[2] = xi[2] + xi[3] * t[2];
                du = k[sj][0] * xj[0] / xj[2] + k[sj][2] - u;
                dv = k[sj][1] * xj[1] / xj[2] + k[sj][3] - v;
                total += 1.0f - beta;
                if (xj[2] > kMinDepth) {
                    accum += (1.0f - beta) * std::sqrt(du * du + dv * dv);
                    valid += 1.0f - beta;
                }
            }
        }
        dist[edge] = valid / (total + 1e-8f) < 0.75f ? 1000.0f : accum / valid;
    }
    return result;
}

torch::Tensor depth_filter_cpu(torch::Tensor poses, torch::Tensor disps, torch::Tensor intrinsics,
                               torch::Tensor inds, torch::Tensor thresh) {
    check_cpu_float(poses, "poses");
    check_cpu_float(disps, "disps");
    check_cpu_float(intrinsics, "intrinsics");
    check_cpu_long(inds, "inds");
    check_cpu_float(thresh, "thresh");
    TORCH_CHECK(inds.numel() == thresh.numel(), "inds and thresh must have the same length");

    auto result = torch::zeros({inds.size(0), disps.size(1), disps.size(2)}, disps.options());
    const auto p = poses.accessor<float, 2>();
    const auto d = disps.accessor<float, 3>();
    const auto k = intrinsics.accessor<float, 1>();
    const auto frames = inds.accessor<int64_t, 1>();
    const auto thresholds = thresh.accessor<float, 1>();
    auto count = result.accessor<float, 3>();
    const float fx = k[0], fy = k[1], cx = k[2], cy = k[3];

    for (int64_t n = 0; n < inds.size(0); ++n) {
        const int64_t i = frames[n];
        for (int neighbor = 0; neighbor < 6; ++neighbor) {
            const int64_t j = neighbor < 3 ? i - neighbor - 1 : i + neighbor - 2;
            if (j < 0 || j >= disps.size(0)) continue;
            float t[3], q[4];
            load_relative_pose(p, i, j, t, q);
            for (int64_t y = 0; y < disps.size(1); ++y) {
                for (int64_t x = 0; x < disps.size(2); ++x) {
                    const float xi[4] = {(static_cast<float>(x) - cx) / fx, (static_cast<float>(y) - cy) / fy,
                                         1.0f, d[i][y][x]};
                    float xj[4];
                    act_se3(t, q, xi, xj);
                    const float uj = fx * xj[0] / xj[2] + cx;
                    const float vj = fy * xj[1] / xj[2] + cy;
                    const float dj = xj[3] / xj[2];
                    const int64_t u0 = static_cast<int64_t>(std::floor(uj));
                    const int64_t v0 = static_cast<int64_t>(std::floor(vj));
                    if (u0 < 0 || v0 < 0 || u0 >= disps.size(2) - 1 || v0 >= disps.size(1) - 1) continue;
                    const float candidates[4] = {d[j][v0][u0], d[j][v0][u0 + 1], d[j][v0 + 1][u0],
                                                 d[j][v0 + 1][u0 + 1]};
                    for (float candidate : candidates) {
                        if (std::abs(1.0f / dj - 1.0f / candidate) < thresholds[n]) {
                            count[n][y][x] += 1.0f;
                            break;
                        }
                    }
                }
            }
        }
    }
    return result;
}

}  // namespace slam_ext

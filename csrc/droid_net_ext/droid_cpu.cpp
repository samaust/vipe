/** CPU correlation kernels matching the DROID-SLAM CUDA extension layouts. */

#include <torch/extension.h>

#include <cmath>
#include <vector>

namespace {

void check_float_cpu(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must have dtype float32 on CPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline bool inside(int64_t y, int64_t x, int64_t height, int64_t width) {
    return y >= 0 && y < height && x >= 0 && x < width;
}

}  // namespace

std::vector<torch::Tensor> corr_index_cpu_forward(torch::Tensor volume, torch::Tensor coords, int radius) {
    check_float_cpu(volume, "volume");
    check_float_cpu(coords, "coords");
    const int64_t rd = 2 * radius + 1;
    auto result = torch::zeros({volume.size(0), rd, rd, volume.size(1), volume.size(2)}, volume.options());
    const auto v = volume.accessor<float, 5>();
    const auto xy = coords.accessor<float, 4>();
    auto out = result.accessor<float, 5>();

    for (int64_t b = 0; b < volume.size(0); ++b)
        for (int64_t y = 0; y < volume.size(1); ++y)
            for (int64_t x = 0; x < volume.size(2); ++x) {
                const float px = xy[b][0][y][x], py = xy[b][1][y][x];
                const int64_t x0 = static_cast<int64_t>(std::floor(px));
                const int64_t y0 = static_cast<int64_t>(std::floor(py));
                const float dx = px - x0, dy = py - y0;
                for (int64_t ox = 0; ox < rd; ++ox)
                    for (int64_t oy = 0; oy < rd; ++oy) {
                        float value = 0.0f;
                        for (int64_t sx = 0; sx < 2; ++sx)
                            for (int64_t sy = 0; sy < 2; ++sy) {
                                const int64_t xx = x0 - radius + ox + sx;
                                const int64_t yy = y0 - radius + oy + sy;
                                if (!inside(yy, xx, volume.size(3), volume.size(4))) continue;
                                const float wx = sx ? dx : 1.0f - dx;
                                const float wy = sy ? dy : 1.0f - dy;
                                value += wx * wy * v[b][y][x][yy][xx];
                            }
                        out[b][ox][oy][y][x] = value;
                    }
            }
    return {result};
}

std::vector<torch::Tensor> corr_index_cpu_backward(torch::Tensor volume, torch::Tensor coords,
                                                    torch::Tensor corr_grad, int radius) {
    check_float_cpu(volume, "volume");
    check_float_cpu(coords, "coords");
    check_float_cpu(corr_grad, "corr_grad");
    const int64_t rd = 2 * radius + 1;
    auto result = torch::zeros_like(volume);
    const auto xy = coords.accessor<float, 4>();
    const auto grad = corr_grad.accessor<float, 5>();
    auto out = result.accessor<float, 5>();

    for (int64_t b = 0; b < volume.size(0); ++b)
        for (int64_t y = 0; y < volume.size(1); ++y)
            for (int64_t x = 0; x < volume.size(2); ++x) {
                const float px = xy[b][0][y][x], py = xy[b][1][y][x];
                const int64_t x0 = static_cast<int64_t>(std::floor(px));
                const int64_t y0 = static_cast<int64_t>(std::floor(py));
                const float dx = px - x0, dy = py - y0;
                for (int64_t ox = 0; ox < rd; ++ox)
                    for (int64_t oy = 0; oy < rd; ++oy)
                        for (int64_t sx = 0; sx < 2; ++sx)
                            for (int64_t sy = 0; sy < 2; ++sy) {
                                const int64_t xx = x0 - radius + ox + sx;
                                const int64_t yy = y0 - radius + oy + sy;
                                if (!inside(yy, xx, volume.size(3), volume.size(4))) continue;
                                const float wx = sx ? dx : 1.0f - dx;
                                const float wy = sy ? dy : 1.0f - dy;
                                out[b][y][x][yy][xx] += wx * wy * grad[b][ox][oy][y][x];
                            }
            }
    return {result};
}

std::vector<torch::Tensor> altcorr_cpu_forward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                               int radius) {
    check_float_cpu(fmap1, "fmap1");
    check_float_cpu(fmap2, "fmap2");
    check_float_cpu(coords, "coords");
    const int64_t rd = 2 * radius + 1;
    auto result = torch::zeros({coords.size(0), coords.size(1), rd * rd, coords.size(2), coords.size(3)},
                               fmap1.options());
    const auto f1 = fmap1.accessor<float, 4>();
    const auto f2 = fmap2.accessor<float, 4>();
    const auto xy = coords.accessor<float, 5>();
    auto out = result.accessor<float, 5>();

    for (int64_t b = 0; b < coords.size(0); ++b)
        for (int64_t n = 0; n < coords.size(1); ++n)
            for (int64_t y = 0; y < coords.size(2); ++y)
                for (int64_t x = 0; x < coords.size(3); ++x) {
                    const float px = xy[b][n][y][x][0], py = xy[b][n][y][x][1];
                    const int64_t x0 = static_cast<int64_t>(std::floor(px));
                    const int64_t y0 = static_cast<int64_t>(std::floor(py));
                    const float dx = px - x0, dy = py - y0;
                    for (int64_t ox = 0; ox < rd; ++ox)
                        for (int64_t oy = 0; oy < rd; ++oy) {
                            float corr = 0.0f;
                            for (int64_t sx = 0; sx < 2; ++sx)
                                for (int64_t sy = 0; sy < 2; ++sy) {
                                    const int64_t xx = x0 - radius + ox + sx;
                                    const int64_t yy = y0 - radius + oy + sy;
                                    if (!inside(yy, xx, fmap2.size(1), fmap2.size(2))) continue;
                                    float dot = 0.0f;
                                    for (int64_t c = 0; c < fmap1.size(3); ++c)
                                        dot += f1[b][y][x][c] * f2[b][yy][xx][c];
                                    corr += (sx ? dx : 1.0f - dx) * (sy ? dy : 1.0f - dy) * dot;
                                }
                            out[b][n][ox * rd + oy][y][x] = corr;
                        }
                }
    return {result};
}

std::vector<torch::Tensor> altcorr_cpu_backward(torch::Tensor fmap1, torch::Tensor fmap2, torch::Tensor coords,
                                                torch::Tensor corr_grad, int radius) {
    check_float_cpu(fmap1, "fmap1");
    check_float_cpu(fmap2, "fmap2");
    check_float_cpu(coords, "coords");
    check_float_cpu(corr_grad, "corr_grad");
    const int64_t rd = 2 * radius + 1;
    auto grad1 = torch::zeros_like(fmap1);
    auto grad2 = torch::zeros_like(fmap2);
    auto grad_coords = torch::zeros_like(coords);  // CUDA implementation also leaves this zero.
    const auto f1 = fmap1.accessor<float, 4>();
    const auto f2 = fmap2.accessor<float, 4>();
    const auto xy = coords.accessor<float, 5>();
    const auto grad = corr_grad.accessor<float, 5>();
    auto g1 = grad1.accessor<float, 4>();
    auto g2 = grad2.accessor<float, 4>();

    for (int64_t b = 0; b < coords.size(0); ++b)
        for (int64_t n = 0; n < coords.size(1); ++n)
            for (int64_t y = 0; y < coords.size(2); ++y)
                for (int64_t x = 0; x < coords.size(3); ++x) {
                    const float px = xy[b][n][y][x][0], py = xy[b][n][y][x][1];
                    const int64_t x0 = static_cast<int64_t>(std::floor(px));
                    const int64_t y0 = static_cast<int64_t>(std::floor(py));
                    const float dx = px - x0, dy = py - y0;
                    for (int64_t ox = 0; ox < rd; ++ox)
                        for (int64_t oy = 0; oy < rd; ++oy)
                            for (int64_t sx = 0; sx < 2; ++sx)
                                for (int64_t sy = 0; sy < 2; ++sy) {
                                    const int64_t xx = x0 - radius + ox + sx;
                                    const int64_t yy = y0 - radius + oy + sy;
                                    if (!inside(yy, xx, fmap2.size(1), fmap2.size(2))) continue;
                                    const float weight = (sx ? dx : 1.0f - dx) * (sy ? dy : 1.0f - dy);
                                    const float upstream = grad[b][n][ox * rd + oy][y][x] * weight;
                                    for (int64_t c = 0; c < fmap1.size(3); ++c) {
                                        g1[b][y][x][c] += upstream * f2[b][yy][xx][c];
                                        g2[b][yy][xx][c] += upstream * f1[b][y][x][c];
                                    }
                                }
                }
    return {grad1, grad2, grad_coords};
}

std::vector<torch::Tensor> altcorr_index_cpu_forward(torch::Tensor fmap1,
                                                     std::vector<torch::Tensor> fmap2_pyramid,
                                                     torch::Tensor coords, torch::Tensor ii, torch::Tensor jj,
                                                     int radius) {
    check_float_cpu(fmap1, "fmap1");
    check_float_cpu(coords, "coords");
    TORCH_CHECK(ii.device().is_cpu() && jj.device().is_cpu(), "ii and jj must be CPU tensors");
    const int64_t rd = 2 * radius + 1;
    auto result = torch::zeros({coords.size(0), coords.size(1),
                                static_cast<int64_t>(fmap2_pyramid.size()) * rd * rd, coords.size(2), coords.size(3)},
                               fmap1.options().dtype(torch::kFloat32));
    const auto f1 = fmap1.accessor<float, 5>();
    const auto xy = coords.accessor<float, 5>();
    const auto src = ii.accessor<int64_t, 1>();
    const auto dst = jj.accessor<int64_t, 1>();
    auto out = result.accessor<float, 5>();

    for (int64_t level = 0; level < static_cast<int64_t>(fmap2_pyramid.size()); ++level) {
        check_float_cpu(fmap2_pyramid[level], "fmap2 pyramid level");
        const auto f2 = fmap2_pyramid[level].accessor<float, 5>();
        const float scale = static_cast<float>(int64_t{1} << level);
        for (int64_t b = 0; b < coords.size(0); ++b)
            for (int64_t m = 0; m < coords.size(1); ++m)
                for (int64_t y = 0; y < coords.size(2); ++y)
                    for (int64_t x = 0; x < coords.size(3); ++x) {
                        const float px = xy[b][m][y][x][0] / scale, py = xy[b][m][y][x][1] / scale;
                        const int64_t x0 = static_cast<int64_t>(std::floor(px));
                        const int64_t y0 = static_cast<int64_t>(std::floor(py));
                        const float dx = px - x0, dy = py - y0;
                        for (int64_t ox = 0; ox < rd; ++ox)
                            for (int64_t oy = 0; oy < rd; ++oy) {
                                float corr = 0.0f;
                                for (int64_t sx = 0; sx < 2; ++sx)
                                    for (int64_t sy = 0; sy < 2; ++sy) {
                                        const int64_t xx = x0 - radius + ox + sx;
                                        const int64_t yy = y0 - radius + oy + sy;
                                        if (!inside(yy, xx, fmap2_pyramid[level].size(2),
                                                    fmap2_pyramid[level].size(3)))
                                            continue;
                                        float dot = 0.0f;
                                        for (int64_t c = 0; c < fmap1.size(4); ++c)
                                            dot += f1[b][src[m]][y][x][c] * f2[b][dst[m]][yy][xx][c];
                                        corr += (sx ? dx : 1.0f - dx) * (sy ? dy : 1.0f - dy) * dot;
                                    }
                                out[b][m][level * rd * rd + ox * rd + oy][y][x] = corr;
                            }
                    }
    }
    return {result};
}

/*
 * Intel XPU (SYCL) Kernels for GRT (Geometric Rotation Transformer)
 *
 * Implements the SYCL equivalents of all GRT CUDA/ROCm kernels:
 *   - givens_forward / givens_backward
 *   - givens_mor_forward / givens_mor_backward
 *   - givens_coupling_forward / givens_coupling_backward
 *   - clifford_coupling_forward / clifford_coupling_backward
 *   - fused_embedding_forward / fused_embedding_backward
 *   - fused_norm_forward / fused_norm_backward
 *   - fused_gelu_forward / fused_gelu_backward
 *   - fused_mixer_forward / fused_mixer_backward
 *   - fused_zloss_forward / fused_zloss_backward
 */

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int SHMEM_BANK_SIZE = 32;
constexpr int COUPLING_TPB = 16;
constexpr int MIXER_TPB = 16;

inline int shmem_idx(int idx) {
    return idx + (idx / SHMEM_BANK_SIZE) * 4;
}

inline float softplus_stable_sycl(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return sycl::exp(x);
    return sycl::log1p(sycl::exp(x));
}

inline void compute_pair_indices_sycl(
    int pair_idx, int shift_val, int D,
    int& p0, int& p1)
{
    if (shift_val >= 0) {
        int d_mask = D - 1;
        p0 = (2 * pair_idx - shift_val + D) & d_mask;
        p1 = (2 * pair_idx + 1 - shift_val + D) & d_mask;
    } else {
        int stride = -shift_val;
        int stride_mask = stride - 1;
        int block_start = (pair_idx & ~stride_mask) << 1;
        int in_block = pair_idx & stride_mask;
        p0 = block_start + in_block;
        p1 = block_start + in_block + stride;
    }
}

template <typename DataT>
inline float load_elem(const DataT* ptr, int idx) {
    return static_cast<float>(ptr[idx]);
}

template <typename DataT>
inline void store_elem(DataT* ptr, int idx, float val) {
    ptr[idx] = static_cast<DataT>(val);
}

template <typename DataT> struct k_givens_fwd;
template <typename DataT> struct k_givens_bwd;
template <typename DataT> struct k_givens_mor_fwd;
template <typename DataT> struct k_givens_mor_bwd;

template <typename DataT>
void givens_fwd_sycl_impl(
    sycl::queue& q,
    const DataT* x,
    const float* angles,
    const float* alpha,
    const float* scales,
    DataT* out,
    const int* shifts,
    int N, int H, int D, int K)
{
    int blocks = N * H;
    if (blocks <= 0 || D <= 0) return;
    int half_D = D / 2;
    int threads = std::max(1, std::min(half_D, 256));
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;

    sycl::range<1> global_size(static_cast<size_t>(blocks) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(padded_D), h);

        h.parallel_for<k_givens_fwd<DataT>>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int block_id = item.get_group(0);
                int n = block_id / H;
                int hh = block_id % H;
                if (n >= N) return;

                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);
                float* s_ptr = smem.get_multi_ptr<sycl::access::decorated::no>().get();

                int x_offset = (n * H + hh) * D;
                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    s_ptr[shmem_idx(2 * pair_idx)]     = load_elem(x, x_offset + 2 * pair_idx);
                    s_ptr[shmem_idx(2 * pair_idx + 1)] = load_elem(x, x_offset + 2 * pair_idx + 1);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < K; k++) {
                    int shift = shifts[k];
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        int p0, p1;
                        compute_pair_indices_sycl(pair_idx, shift, D, p0, p1);

                        float x0 = s_ptr[shmem_idx(p0)];
                        float x1 = s_ptr[shmem_idx(p1)];

                        int cs_offset = (k * H + hh) * half_D + pair_idx;
                        float phase = sycl::atan2(x1, x0);
                        float theta_eff = angles[cs_offset] + alpha[cs_offset] * phase;
                        float c = sycl::cos(theta_eff);
                        float s = sycl::sin(theta_eff);

                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        int sc_offset = (k * H + hh) * D;
                        r0 *= scales[sc_offset + 2 * pair_idx];
                        r1 *= scales[sc_offset + 2 * pair_idx + 1];

                        s_ptr[shmem_idx(p0)] = r0;
                        s_ptr[shmem_idx(p1)] = r1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    store_elem(out, x_offset + 2 * pair_idx,     s_ptr[shmem_idx(2 * pair_idx)]);
                    store_elem(out, x_offset + 2 * pair_idx + 1, s_ptr[shmem_idx(2 * pair_idx + 1)]);
                }
            });
    });
}

template <typename DataT>
void givens_bwd_sycl_impl(
    sycl::queue& q,
    const DataT* grad_out,
    const DataT* x_in,
    const float* angles,
    const float* alpha,
    const float* scales,
    DataT* grad_x,
    float* grad_angles,
    float* grad_alpha,
    float* grad_scales,
    const int* shifts,
    float* save_buf,
    int N, int H, int D, int K)
{
    int blocks = N * H;
    if (blocks <= 0 || D <= 0) return;
    int half_D = D / 2;
    int threads = std::max(1, std::min(half_D, 256));
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;

    sycl::range<1> global_size(static_cast<size_t>(blocks) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(padded_D), h);

        h.parallel_for<k_givens_bwd<DataT>>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int block_id = item.get_group(0);
                int n = block_id / H;
                int hh = block_id % H;
                if (n >= N) return;

                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);
                float* work = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* save = save_buf + static_cast<long long>(block_id) * K * D;
                int x_offset = (n * H + hh) * D;

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    work[shmem_idx(2 * pair_idx)]     = load_elem(x_in, x_offset + 2 * pair_idx);
                    work[shmem_idx(2 * pair_idx + 1)] = load_elem(x_in, x_offset + 2 * pair_idx + 1);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < K; k++) {
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        save[k * D + 2 * pair_idx]     = work[shmem_idx(2 * pair_idx)];
                        save[k * D + 2 * pair_idx + 1] = work[shmem_idx(2 * pair_idx + 1)];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    int shift = shifts[k];
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        int p0, p1;
                        compute_pair_indices_sycl(pair_idx, shift, D, p0, p1);

                        float x0 = work[shmem_idx(p0)];
                        float x1 = work[shmem_idx(p1)];

                        int cs_offset = (k * H + hh) * half_D + pair_idx;
                        float phase = sycl::atan2(x1, x0);
                        float theta_eff = angles[cs_offset] + alpha[cs_offset] * phase;
                        float c = sycl::cos(theta_eff);
                        float s = sycl::sin(theta_eff);

                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        int sc_offset = (k * H + hh) * D;
                        r0 *= scales[sc_offset + 2 * pair_idx];
                        r1 *= scales[sc_offset + 2 * pair_idx + 1];

                        work[shmem_idx(p0)] = r0;
                        work[shmem_idx(p1)] = r1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    work[shmem_idx(2 * pair_idx)]     = load_elem(grad_out, x_offset + 2 * pair_idx);
                    work[shmem_idx(2 * pair_idx + 1)] = load_elem(grad_out, x_offset + 2 * pair_idx + 1);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = K - 1; k >= 0; k--) {
                    int shift = shifts[k];
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        int p0, p1;
                        compute_pair_indices_sycl(pair_idx, shift, D, p0, p1);

                        float g_out0 = work[shmem_idx(p0)];
                        float g_out1 = work[shmem_idx(p1)];

                        float x0 = save[k * D + p0];
                        float x1 = save[k * D + p1];

                        int cs_offset = (k * H + hh) * half_D + pair_idx;
                        float phase = sycl::atan2(x1, x0);
                        float a_val = alpha[cs_offset];
                        float theta_eff = angles[cs_offset] + a_val * phase;
                        float c = sycl::cos(theta_eff);
                        float s = sycl::sin(theta_eff);

                        int sc_offset = (k * H + hh) * D;
                        float s0 = scales[sc_offset + 2 * pair_idx];
                        float s1 = scales[sc_offset + 2 * pair_idx + 1];

                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                gs0(grad_scales[sc_offset + 2 * pair_idx]);
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                gs1(grad_scales[sc_offset + 2 * pair_idx + 1]);
                            gs0.fetch_add(r0 * g_out0);
                            gs1.fetch_add(r1 * g_out1);
                        }

                        float g_r0 = g_out0 * s0;
                        float g_r1 = g_out1 * s1;
                        float g_theta = g_r0 * (-r1) + g_r1 * r0;

                        {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                ga(grad_angles[cs_offset]);
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                gal(grad_alpha[cs_offset]);
                            ga.fetch_add(g_theta);
                            gal.fetch_add(g_theta * phase);
                        }

                        float g_x0 = g_r0 * c + g_r1 * s;
                        float g_x1 = -g_r0 * s + g_r1 * c;

                        float norm_sq = x0 * x0 + x1 * x1;
                        if (norm_sq > 1e-12f) {
                            float inv_norm_sq = a_val / norm_sq;
                            g_x0 += g_theta * (-x1 * inv_norm_sq);
                            g_x1 += g_theta * (x0 * inv_norm_sq);
                        }

                        work[shmem_idx(p0)] = g_x0;
                        work[shmem_idx(p1)] = g_x1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    store_elem(grad_x, x_offset + 2 * pair_idx,     work[shmem_idx(2 * pair_idx)]);
                    store_elem(grad_x, x_offset + 2 * pair_idx + 1, work[shmem_idx(2 * pair_idx + 1)]);
                }
            });
    });
}

template <typename DataT>
void givens_mor_fwd_sycl_impl(
    sycl::queue& q,
    const DataT* x,
    const float* expert_angles,
    const float* expert_scales,
    const float* router_weights,
    DataT* out,
    const int* shifts,
    int N, int H, int D, int K, int E)
{
    int blocks = N * H;
    if (blocks <= 0 || D <= 0) return;
    int half_D = D / 2;
    int threads = std::max(1, std::min(half_D, 256));
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_total = padded_D + E;

    sycl::range<1> global_size(static_cast<size_t>(blocks) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_total), h);

        h.parallel_for<k_givens_mor_fwd<DataT>>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int block_id = item.get_group(0);
                int n = block_id / H;
                int hh = block_id % H;
                if (n >= N) return;

                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);
                float* s_ptr = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* work = s_ptr;
                float* rw = s_ptr + padded_D;

                int x_offset = (n * H + hh) * D;
                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    work[shmem_idx(2 * pair_idx)]     = load_elem(x, x_offset + 2 * pair_idx);
                    work[shmem_idx(2 * pair_idx + 1)] = load_elem(x, x_offset + 2 * pair_idx + 1);
                }
                for (int e = tid; e < E; e += block_dim) {
                    rw[e] = router_weights[n * E + e];
                }
                item.barrier(sycl::access::fence_space::local_space);

                int kh_stride = K * H * half_D;
                int kh_stride_s = K * H * D;

                for (int k = 0; k < K; k++) {
                    int shift = shifts[k];
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        int p0, p1;
                        compute_pair_indices_sycl(pair_idx, shift, D, p0, p1);

                        float x0 = work[shmem_idx(p0)];
                        float x1 = work[shmem_idx(p1)];

                        int ang_base = (k * H + hh) * half_D + pair_idx;
                        float theta = 0.0f;
                        for (int e = 0; e < E; e++) {
                            theta += rw[e] * expert_angles[e * kh_stride + ang_base];
                        }

                        float c = sycl::cos(theta);
                        float s = sycl::sin(theta);
                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        int sc_base = (k * H + hh) * D;
                        float s0 = 0.0f, s1 = 0.0f;
                        for (int e = 0; e < E; e++) {
                            int sc_off = e * kh_stride_s + sc_base;
                            s0 += rw[e] * expert_scales[sc_off + 2 * pair_idx];
                            s1 += rw[e] * expert_scales[sc_off + 2 * pair_idx + 1];
                        }

                        work[shmem_idx(p0)] = r0 * s0;
                        work[shmem_idx(p1)] = r1 * s1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    store_elem(out, x_offset + 2 * pair_idx,     work[shmem_idx(2 * pair_idx)]);
                    store_elem(out, x_offset + 2 * pair_idx + 1, work[shmem_idx(2 * pair_idx + 1)]);
                }
            });
    });
}

template <typename DataT>
void givens_mor_bwd_sycl_impl(
    sycl::queue& q,
    const DataT* grad_out,
    const DataT* x_in,
    const float* expert_angles,
    const float* expert_scales,
    const float* router_weights,
    DataT* grad_x,
    float* grad_expert_angles,
    float* grad_expert_scales,
    float* grad_router,
    const int* shifts,
    float* save_buf,
    int N, int H, int D, int K, int E)
{
    int blocks = N * H;
    if (blocks <= 0 || D <= 0) return;
    int half_D = D / 2;
    int threads = std::max(1, std::min(half_D, 256));
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int smem_total = padded_D + 2 * E;

    sycl::range<1> global_size(static_cast<size_t>(blocks) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_total), h);

        h.parallel_for<k_givens_mor_bwd<DataT>>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int block_id = item.get_group(0);
                int n = block_id / H;
                int hh = block_id % H;
                if (n >= N) return;

                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);
                float* s_ptr = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* work = s_ptr;
                float* rw = s_ptr + padded_D;
                float* grw = rw + E;

                float* save = save_buf + static_cast<long long>(block_id) * K * D;
                int x_offset = (n * H + hh) * D;
                int kh_stride = K * H * half_D;
                int kh_stride_s = K * H * D;

                for (int e = tid; e < E; e += block_dim) {
                    rw[e] = router_weights[n * E + e];
                    grw[e] = 0.0f;
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    work[shmem_idx(2 * pair_idx)]     = load_elem(x_in, x_offset + 2 * pair_idx);
                    work[shmem_idx(2 * pair_idx + 1)] = load_elem(x_in, x_offset + 2 * pair_idx + 1);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < K; k++) {
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        save[k * D + 2 * pair_idx]     = work[shmem_idx(2 * pair_idx)];
                        save[k * D + 2 * pair_idx + 1] = work[shmem_idx(2 * pair_idx + 1)];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    int shift = shifts[k];
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        int p0, p1;
                        compute_pair_indices_sycl(pair_idx, shift, D, p0, p1);

                        float x0 = work[shmem_idx(p0)];
                        float x1 = work[shmem_idx(p1)];

                        int ang_base = (k * H + hh) * half_D + pair_idx;
                        float theta = 0.0f;
                        for (int e = 0; e < E; e++) {
                            theta += rw[e] * expert_angles[e * kh_stride + ang_base];
                        }
                        float c = sycl::cos(theta);
                        float s = sycl::sin(theta);
                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        int sc_base = (k * H + hh) * D;
                        float s0 = 0.0f, s1 = 0.0f;
                        for (int e = 0; e < E; e++) {
                            int sc_off = e * kh_stride_s + sc_base;
                            s0 += rw[e] * expert_scales[sc_off + 2 * pair_idx];
                            s1 += rw[e] * expert_scales[sc_off + 2 * pair_idx + 1];
                        }

                        work[shmem_idx(p0)] = r0 * s0;
                        work[shmem_idx(p1)] = r1 * s1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    work[shmem_idx(2 * pair_idx)]     = load_elem(grad_out, x_offset + 2 * pair_idx);
                    work[shmem_idx(2 * pair_idx + 1)] = load_elem(grad_out, x_offset + 2 * pair_idx + 1);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = K - 1; k >= 0; k--) {
                    int shift = shifts[k];
                    for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                        int p0, p1;
                        compute_pair_indices_sycl(pair_idx, shift, D, p0, p1);

                        float g_out0 = work[shmem_idx(p0)];
                        float g_out1 = work[shmem_idx(p1)];

                        float x0 = save[k * D + p0];
                        float x1 = save[k * D + p1];

                        int ang_base = (k * H + hh) * half_D + pair_idx;
                        float theta = 0.0f;
                        for (int e = 0; e < E; e++) {
                            theta += rw[e] * expert_angles[e * kh_stride + ang_base];
                        }
                        float c = sycl::cos(theta);
                        float s = sycl::sin(theta);
                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        int sc_base = (k * H + hh) * D;
                        float s0_val = 0.0f, s1_val = 0.0f;
                        for (int e = 0; e < E; e++) {
                            int sc_off = e * kh_stride_s + sc_base;
                            s0_val += rw[e] * expert_scales[sc_off + 2 * pair_idx];
                            s1_val += rw[e] * expert_scales[sc_off + 2 * pair_idx + 1];
                        }

                        float g_r0 = g_out0 * s0_val;
                        float g_r1 = g_out1 * s1_val;
                        float g_s0_blend = g_out0 * r0;
                        float g_s1_blend = g_out1 * r1;

                        for (int e = 0; e < E; e++) {
                            int sc_off = e * kh_stride_s + sc_base;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                ges0(grad_expert_scales[sc_off + 2 * pair_idx]);
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                ges1(grad_expert_scales[sc_off + 2 * pair_idx + 1]);
                            ges0.fetch_add(rw[e] * g_s0_blend);
                            ges1.fetch_add(rw[e] * g_s1_blend);
                        }

                        float g_theta = g_r0 * (-r1) + g_r1 * r0;
                        for (int e = 0; e < E; e++) {
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                gea(grad_expert_angles[e * kh_stride + ang_base]);
                            gea.fetch_add(rw[e] * g_theta);
                        }

                        for (int e = 0; e < E; e++) {
                            float gr_e = g_theta * expert_angles[e * kh_stride + ang_base];
                            int sc_off = e * kh_stride_s + sc_base;
                            gr_e += g_s0_blend * expert_scales[sc_off + 2 * pair_idx];
                            gr_e += g_s1_blend * expert_scales[sc_off + 2 * pair_idx + 1];
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>
                                grw_ref(grw[e]);
                            grw_ref.fetch_add(gr_e);
                        }

                        float g_x0 = g_r0 * c + g_r1 * s;
                        float g_x1 = -g_r0 * s + g_r1 * c;
                        work[shmem_idx(p0)] = g_x0;
                        work[shmem_idx(p1)] = g_x1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int pair_idx = tid; pair_idx < half_D; pair_idx += block_dim) {
                    store_elem(grad_x, x_offset + 2 * pair_idx,     work[shmem_idx(2 * pair_idx)]);
                    store_elem(grad_x, x_offset + 2 * pair_idx + 1, work[shmem_idx(2 * pair_idx + 1)]);
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int e = tid; e < E; e += block_dim) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        gr_ref(grad_router[n * E + e]);
                    gr_ref.fetch_add(grw[e]);
                }
            });
    });
}

} // namespace

extern "C" void launch_givens_fwd(
    const void* x, const float* angles, const float* alpha,
    const float* scales, void* out, const int* shifts,
    int N, int H, int D, int K, int use_half)
{
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    if (use_half) {
        givens_fwd_sycl_impl<at::Half>(
            q, static_cast<const at::Half*>(x), angles, alpha, scales,
            static_cast<at::Half*>(out), shifts, N, H, D, K);
    } else {
        givens_fwd_sycl_impl<float>(
            q, static_cast<const float*>(x), angles, alpha, scales,
            static_cast<float*>(out), shifts, N, H, D, K);
    }
}

extern "C" void launch_givens_bwd(
    const void* grad_out, const void* x_in,
    const float* angles, const float* alpha, const float* scales,
    void* grad_x, float* grad_angles, float* grad_alpha, float* grad_scales,
    const int* shifts, float* save_buf,
    int N, int H, int D, int K, int use_half)
{
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    if (use_half) {
        givens_bwd_sycl_impl<at::Half>(
            q, static_cast<const at::Half*>(grad_out), static_cast<const at::Half*>(x_in),
            angles, alpha, scales, static_cast<at::Half*>(grad_x),
            grad_angles, grad_alpha, grad_scales, shifts, save_buf, N, H, D, K);
    } else {
        givens_bwd_sycl_impl<float>(
            q, static_cast<const float*>(grad_out), static_cast<const float*>(x_in),
            angles, alpha, scales, static_cast<float*>(grad_x),
            grad_angles, grad_alpha, grad_scales, shifts, save_buf, N, H, D, K);
    }
}

extern "C" void launch_givens_mor_fwd(
    const void* x, const float* expert_angles, const float* expert_scales,
    const float* router_weights, void* out, const int* shifts,
    int N, int H, int D, int K, int E, int use_half)
{
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    if (use_half) {
        givens_mor_fwd_sycl_impl<at::Half>(
            q, static_cast<const at::Half*>(x), expert_angles, expert_scales,
            router_weights, static_cast<at::Half*>(out), shifts, N, H, D, K, E);
    } else {
        givens_mor_fwd_sycl_impl<float>(
            q, static_cast<const float*>(x), expert_angles, expert_scales,
            router_weights, static_cast<float*>(out), shifts, N, H, D, K, E);
    }
}

extern "C" void launch_givens_mor_bwd(
    const void* grad_out, const void* x_in,
    const float* expert_angles, const float* expert_scales,
    const float* router_weights,
    void* grad_x, float* grad_expert_angles, float* grad_expert_scales,
    float* grad_router, const int* shifts, float* save_buf,
    int N, int H, int D, int K, int E, int use_half)
{
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    if (use_half) {
        givens_mor_bwd_sycl_impl<at::Half>(
            q, static_cast<const at::Half*>(grad_out), static_cast<const at::Half*>(x_in),
            expert_angles, expert_scales, router_weights,
            static_cast<at::Half*>(grad_x), grad_expert_angles, grad_expert_scales,
            grad_router, shifts, save_buf, N, H, D, K, E);
    } else {
        givens_mor_bwd_sycl_impl<float>(
            q, static_cast<const float*>(grad_out), static_cast<const float*>(x_in),
            expert_angles, expert_scales, router_weights,
            static_cast<float*>(grad_x), grad_expert_angles, grad_expert_scales,
            grad_router, shifts, save_buf, N, H, D, K, E);
    }
}

extern "C" void launch_givens_coupling_fwd(
    const float* x, const float* angles,
    const float* coupling, const float* scales, const int* shifts, float* out,
    int N, int H, int D, int K, int B)
{
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int half_D = D / 2;
    int f_feat = 4;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int coupling_size = f_feat * half_D * half_D;
    int TPB = std::max(1, std::min(COUPLING_TPB, 256 / std::max(1, half_D)));

    int n_tiles = (N + TPB - 1) / TPB;
    int n_blocks = B * n_tiles * H;
    if (n_blocks <= 0) return;

    int smem_total = TPB * padded_D
                   + TPB * half_D
                   + TPB * half_D
                   + TPB * f_feat * half_D;

    sycl::range<2> global_size(static_cast<size_t>(n_blocks) * TPB, half_D);
    sycl::range<2> local_size(TPB, half_D);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_total), h);

        h.parallel_for<class k_givens_coupling_fwd>(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) {
                const int token_slot = item.get_local_id(0);
                const int p          = item.get_local_id(1);
                const int block_id   = item.get_group(0);

                const int b      = block_id / (n_tiles * H);
                const int rest   = block_id % (n_tiles * H);
                const int n_tile = rest / H;
                const int hh     = rest % H;

                const int my_n = n_tile * TPB + token_slot;
                const bool valid = (my_n < N);

                float* work_base     = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* smem_x0_base  = work_base + TPB * padded_D;
                float* smem_x1_base  = smem_x0_base + TPB * half_D;
                float* feat_base     = smem_x1_base + TPB * half_D;

                float* my_work     = work_base + token_slot * padded_D;
                float* my_x0       = smem_x0_base + token_slot * half_D;
                float* my_x1       = smem_x1_base + token_slot * half_D;
                float* my_features = feat_base + token_slot * f_feat * half_D;

                if (valid) {
                    const int x_offset = (my_n * H + hh) * D;
                    my_work[shmem_idx(2 * p)]     = x[x_offset + 2 * p];
                    my_work[shmem_idx(2 * p + 1)] = x[x_offset + 2 * p + 1];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < K; k++) {
                    const int cp_base = ((b * K + k) * H + hh) * coupling_size;
                    const float* coupling_tile = coupling + cp_base;

                    int p0_perm, p1_perm;
                    compute_pair_indices_sycl(p, shifts[k], D, p0_perm, p1_perm);
                    if (valid) {
                        my_x0[p] = my_work[shmem_idx(p0_perm)];
                        my_x1[p] = my_work[shmem_idx(p1_perm)];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (valid) {
                        const float x0 = my_x0[p];
                        const float x1 = my_x1[p];
                        const int next = (p + 1) & (half_D - 1);
                        const float x0n = my_x0[next];
                        const float x1n = my_x1[next];

                        my_features[p]              = x0 * x0 + x1 * x1;
                        my_features[half_D + p]     = x0;
                        my_features[2 * half_D + p] = x1;
                        my_features[3 * half_D + p] = x0 * x1n - x1 * x0n;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float saved_x0 = 0.0f, saved_x1 = 0.0f;
                    if (valid) {
                        saved_x0 = my_x0[p];
                        saved_x1 = my_x1[p];
                        float sq = 0.0f;
                        sq += my_features[p]              * my_features[p];
                        sq += my_features[half_D + p]     * my_features[half_D + p];
                        sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
                        sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
                        my_x0[p] = sq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = half_D / 2; stride > 0; stride >>= 1) {
                        if (valid && p < stride)
                            my_x0[p] += my_x0[p + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (valid) {
                        float sq = my_x0[0];
                        const float inv_norm = sycl::rsqrt(sq + 1e-6f);
                        my_features[p]              *= inv_norm;
                        my_features[half_D + p]     *= inv_norm;
                        my_features[2 * half_D + p] *= inv_norm;
                        my_features[3 * half_D + p] *= inv_norm;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (valid) {
                        const int total = f_feat * half_D;
                        float modulation = 0.0f;
                        for (int j = 0; j < total; j++)
                            modulation += my_features[j] * coupling_tile[j * half_D + p];

                        const int a_off = ((b * K + k) * H + hh) * half_D + p;
                        const float correction = sycl::tanh(modulation);
                        const float c_base = sycl::cos(angles[a_off]);
                        const float s_base = sycl::sin(angles[a_off]);
                        const float c_corr = sycl::cos(correction);
                        const float s_corr = sycl::sin(correction);
                        const float c = c_base * c_corr - s_base * s_corr;
                        const float s = s_base * c_corr + c_base * s_corr;

                        const float x0 = saved_x0;
                        const float x1 = saved_x1;
                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        const int s_off = ((b * K + k) * H + hh) * D;
                        r0 *= scales[s_off + 2 * p];
                        r1 *= scales[s_off + 2 * p + 1];

                        my_work[shmem_idx(p0_perm)] = r0;
                        my_work[shmem_idx(p1_perm)] = r1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (valid) {
                    const int out_offset = ((b * N + my_n) * H + hh) * D;
                    out[out_offset + 2 * p]     = my_work[shmem_idx(2 * p)];
                    out[out_offset + 2 * p + 1] = my_work[shmem_idx(2 * p + 1)];
                }
            });
    });
}

extern "C" void launch_givens_coupling_bwd(
    const float* grad_out, const float* x_in,
    const float* angles,
    const float* coupling, const float* scales,
    const int* shifts,
    float* grad_x, float* grad_angles, float* grad_scales,
    float* feat_buf, float* grad_mod_buf,
    float* save_buf,
    int N, int H, int D, int K, int B)
{
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int half_D = D / 2;
    int f_feat = 4;
    int padded_D = D + (D / SHMEM_BANK_SIZE) * 4;
    int coupling_stride = half_D;
    int coupling_size = f_feat * half_D * half_D;
    int TPB = std::max(1, std::min(COUPLING_TPB, 256 / std::max(1, half_D)));

    int n_tiles = (N + TPB - 1) / TPB;
    int n_blocks = B * n_tiles * H;
    if (n_blocks <= 0) return;

    int smem_total = TPB * padded_D
                   + TPB * half_D
                   + TPB * half_D
                   + TPB * f_feat * half_D
                   + TPB * half_D
                   + TPB * half_D
                   + TPB * half_D
                   + half_D
                   + D;

    sycl::range<2> global_size(static_cast<size_t>(n_blocks) * TPB, half_D);
    sycl::range<2> local_size(TPB, half_D);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_total), h);

        h.parallel_for<class k_givens_coupling_bwd>(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) {
                const int token_slot = item.get_local_id(0);
                const int p          = item.get_local_id(1);
                const int block_id   = item.get_group(0);

                const int b      = block_id / (n_tiles * H);
                const int rest   = block_id % (n_tiles * H);
                const int n_tile = rest / H;
                const int hh     = rest % H;

                const int my_n = n_tile * TPB + token_slot;
                const bool valid = (my_n < N);

                float* smem_ptr          = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* work_base         = smem_ptr;
                float* smem_x0_base      = work_base + TPB * padded_D;
                float* smem_x1_base      = smem_x0_base + TPB * half_D;
                float* feat_base_ptr     = smem_x1_base + TPB * half_D;
                float* grad_mod_base     = feat_base_ptr + TPB * f_feat * half_D;
                float* gx0_rot_base      = grad_mod_base + TPB * half_D;
                float* gx1_rot_base      = gx0_rot_base + TPB * half_D;
                float* grad_angles_accum = gx1_rot_base + TPB * half_D;
                float* grad_scales_accum = grad_angles_accum + half_D;

                float* my_work     = work_base + token_slot * padded_D;
                float* my_x0       = smem_x0_base + token_slot * half_D;
                float* my_x1       = smem_x1_base + token_slot * half_D;
                float* my_features = feat_base_ptr + token_slot * f_feat * half_D;
                float* my_grad_mod = grad_mod_base + token_slot * half_D;
                float* my_gx0_rot  = gx0_rot_base + token_slot * half_D;
                float* my_gx1_rot  = gx1_rot_base + token_slot * half_D;

                const long long save_id = valid ? (static_cast<long long>(b * N + my_n) * H + hh) : 0;
                float* my_save = save_buf + save_id * K * D;
                const int x_offset = valid ? ((my_n * H + hh) * D) : 0;

                if (valid) {
                    my_work[shmem_idx(2 * p)]     = x_in[x_offset + 2 * p];
                    my_work[shmem_idx(2 * p + 1)] = x_in[x_offset + 2 * p + 1];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < K; k++) {
                    if (valid) {
                        my_save[k * D + 2 * p]     = my_work[shmem_idx(2 * p)];
                        my_save[k * D + 2 * p + 1] = my_work[shmem_idx(2 * p + 1)];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const float* coupling_tile = coupling + ((b * K + k) * H + hh) * coupling_size;

                    int p0_perm_p1, p1_perm_p1;
                    compute_pair_indices_sycl(p, shifts[k], D, p0_perm_p1, p1_perm_p1);
                    if (valid) {
                        my_x0[p] = my_work[shmem_idx(p0_perm_p1)];
                        my_x1[p] = my_work[shmem_idx(p1_perm_p1)];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (valid) {
                        const float x0 = my_x0[p];
                        const float x1 = my_x1[p];
                        const int next = (p + 1) & (half_D - 1);
                        my_features[p]              = x0 * x0 + x1 * x1;
                        my_features[half_D + p]     = x0;
                        my_features[2 * half_D + p] = x1;
                        my_features[3 * half_D + p] = x0 * my_x1[next] - x1 * my_x0[next];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float p1_saved_x0 = 0.0f, p1_saved_x1 = 0.0f;
                    if (valid) {
                        p1_saved_x0 = my_x0[p];
                        p1_saved_x1 = my_x1[p];
                        float sq = 0.0f;
                        sq += my_features[p]              * my_features[p];
                        sq += my_features[half_D + p]     * my_features[half_D + p];
                        sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
                        sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
                        my_x0[p] = sq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = half_D / 2; stride > 0; stride >>= 1) {
                        if (valid && p < stride)
                            my_x0[p] += my_x0[p + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (valid) {
                        float sq = my_x0[0];
                        const float inv_norm = sycl::rsqrt(sq + 1e-6f);
                        my_features[p]              *= inv_norm;
                        my_features[half_D + p]     *= inv_norm;
                        my_features[2 * half_D + p] *= inv_norm;
                        my_features[3 * half_D + p] *= inv_norm;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (valid) {
                        const int total = f_feat * half_D;
                        float mod = 0.0f;
                        for (int j = 0; j < total; j++)
                            mod += my_features[j] * coupling_tile[j * coupling_stride + p];

                        const int a_off = ((b * K + k) * H + hh) * half_D + p;
                        const float correction = sycl::tanh(mod);
                        const float c_base = sycl::cos(angles[a_off]);
                        const float s_base = sycl::sin(angles[a_off]);
                        const float c_corr = sycl::cos(correction);
                        const float s_corr = sycl::sin(correction);
                        const float c = c_base * c_corr - s_base * s_corr;
                        const float s = s_base * c_corr + c_base * s_corr;

                        const float x0 = p1_saved_x0;
                        const float x1 = p1_saved_x1;
                        float r0 = x0 * c - x1 * s;
                        float r1 = x0 * s + x1 * c;

                        const int s_off = ((b * K + k) * H + hh) * D;
                        r0 *= scales[s_off + 2 * p];
                        r1 *= scales[s_off + 2 * p + 1];

                        my_work[shmem_idx(p0_perm_p1)] = r0;
                        my_work[shmem_idx(p1_perm_p1)] = r1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (valid) {
                    const int out_offset = ((b * N + my_n) * H + hh) * D;
                    my_work[shmem_idx(2 * p)]     = grad_out[out_offset + 2 * p];
                    my_work[shmem_idx(2 * p + 1)] = grad_out[out_offset + 2 * p + 1];
                }
                item.barrier(sycl::access::fence_space::local_space);

                for (int k = K - 1; k >= 0; k--) {
                    if (token_slot == 0) {
                        grad_angles_accum[p]         = 0.0f;
                        grad_scales_accum[2 * p]     = 0.0f;
                        grad_scales_accum[2 * p + 1] = 0.0f;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const float* coupling_tile = coupling + ((b * K + k) * H + hh) * coupling_size;

                    int p0_perm, p1_perm;
                    compute_pair_indices_sycl(p, shifts[k], D, p0_perm, p1_perm);
                    if (valid) {
                        my_x0[p] = my_save[k * D + p0_perm];
                        my_x1[p] = my_save[k * D + p1_perm];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float phase2_inv_norm = 0.0f;
                    if (valid) {
                        const float x0 = my_x0[p];
                        const float x1 = my_x1[p];
                        const int next = (p + 1) & (half_D - 1);
                        my_features[p]              = x0 * x0 + x1 * x1;
                        my_features[half_D + p]     = x0;
                        my_features[2 * half_D + p] = x1;
                        my_features[3 * half_D + p] = x0 * my_x1[next] - x1 * my_x0[next];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    float p2_saved_x0 = 0.0f, p2_saved_x1 = 0.0f;
                    if (valid) {
                        p2_saved_x0 = my_x0[p];
                        p2_saved_x1 = my_x1[p];
                        float sq = 0.0f;
                        sq += my_features[p]              * my_features[p];
                        sq += my_features[half_D + p]     * my_features[half_D + p];
                        sq += my_features[2 * half_D + p] * my_features[2 * half_D + p];
                        sq += my_features[3 * half_D + p] * my_features[3 * half_D + p];
                        my_x0[p] = sq;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = half_D / 2; stride > 0; stride >>= 1) {
                        if (valid && p < stride)
                            my_x0[p] += my_x0[p + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (valid) {
                        float sq = my_x0[0];
                        phase2_inv_norm = sycl::rsqrt(sq + 1e-6f);
                        my_features[p]              *= phase2_inv_norm;
                        my_features[half_D + p]     *= phase2_inv_norm;
                        my_features[2 * half_D + p] *= phase2_inv_norm;
                        my_features[3 * half_D + p] *= phase2_inv_norm;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (valid) {
                        const float g_o0 = my_work[shmem_idx(p0_perm)];
                        const float g_o1 = my_work[shmem_idx(p1_perm)];

                        const float x0 = p2_saved_x0;
                        const float x1 = p2_saved_x1;

                        const int total = f_feat * half_D;
                        float mod = 0.0f;
                        for (int j = 0; j < total; j++)
                            mod += my_features[j] * coupling_tile[j * coupling_stride + p];

                        const int a_off = ((b * K + k) * H + hh) * half_D + p;
                        const float tanh_mod = sycl::tanh(mod);
                        const float c_base2 = sycl::cos(angles[a_off]);
                        const float s_base2 = sycl::sin(angles[a_off]);
                        const float c_corr2 = sycl::cos(tanh_mod);
                        const float s_corr2 = sycl::sin(tanh_mod);
                        const float c = c_base2 * c_corr2 - s_base2 * s_corr2;
                        const float s = s_base2 * c_corr2 + c_base2 * s_corr2;

                        const float r0 = x0 * c - x1 * s;
                        const float r1 = x0 * s + x1 * c;

                        const int s_off = ((b * K + k) * H + hh) * D;
                        const float sc0 = scales[s_off + 2 * p];
                        const float sc1 = scales[s_off + 2 * p + 1];

                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            gs0_ref(grad_scales_accum[2 * p]);
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            gs1_ref(grad_scales_accum[2 * p + 1]);
                        gs0_ref.fetch_add(g_o0 * r0);
                        gs1_ref.fetch_add(g_o1 * r1);

                        const float g_r0 = g_o0 * sc0;
                        const float g_r1 = g_o1 * sc1;

                        const float g_theta = g_r0 * (-r1) + g_r1 * r0;
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            ga_ref(grad_angles_accum[p]);
                        ga_ref.fetch_add(g_theta);

                        const float g_modulation = g_theta * (1.0f - tanh_mod * tanh_mod);
                        my_grad_mod[p] = g_modulation;

                        my_gx0_rot[p] =  g_r0 * c + g_r1 * s;
                        my_gx1_rot[p] = -g_r0 * s + g_r1 * c;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (valid) {
                        const float g_mod_p = my_grad_mod[p];
                        const long long bkh = static_cast<long long>((b * K + k) * H + hh);
                        const long long feat_off = (bkh * N + my_n) * (f_feat * half_D);
                        for (int jj = 0; jj < f_feat; jj++)
                            feat_buf[feat_off + jj * half_D + p] = my_features[jj * half_D + p];
                        const long long gm_off = (bkh * N + my_n) * half_D;
                        grad_mod_buf[gm_off + p] = g_mod_p;
                    }

                    int prev_idx = (p - 1 + half_D) & (half_D - 1);
                    float gn_S0 = 0.0f, gn_V1 = 0.0f, gn_V2 = 0.0f;
                    float gn_B12 = 0.0f, gn_B12_prev = 0.0f;
                    float fn_S0 = 0.0f, fn_V1 = 0.0f, fn_V2 = 0.0f, fn_B12 = 0.0f;

                    if (valid) {
                        my_x0[p] = p2_saved_x0;
                        my_x1[p] = p2_saved_x1;
                        for (int qq = 0; qq < half_D; qq++) {
                            const float gm_q = my_grad_mod[qq];
                            gn_S0       += gm_q * coupling_tile[p * coupling_stride + qq];
                            gn_V1       += gm_q * coupling_tile[(half_D + p) * coupling_stride + qq];
                            gn_V2       += gm_q * coupling_tile[(2 * half_D + p) * coupling_stride + qq];
                            gn_B12      += gm_q * coupling_tile[(3 * half_D + p) * coupling_stride + qq];
                            gn_B12_prev += gm_q * coupling_tile[(3 * half_D + prev_idx) * coupling_stride + qq];
                        }

                        fn_S0  = my_features[p];
                        fn_V1  = my_features[half_D + p];
                        fn_V2  = my_features[2 * half_D + p];
                        fn_B12 = my_features[3 * half_D + p];

                        float dot_partial = gn_S0 * fn_S0 + gn_V1 * fn_V1
                                          + gn_V2 * fn_V2 + gn_B12 * fn_B12;
                        my_features[p] = dot_partial;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int stride = half_D / 2; stride > 0; stride >>= 1) {
                        if (valid && p < stride)
                            my_features[p] += my_features[p + stride];
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                    if (valid) {
                        float dot_partial = my_features[0];
                        const float g_S0  = (gn_S0  - fn_S0  * dot_partial) * phase2_inv_norm;
                        const float g_V1  = (gn_V1  - fn_V1  * dot_partial) * phase2_inv_norm;
                        const float g_V2  = (gn_V2  - fn_V2  * dot_partial) * phase2_inv_norm;
                        const float g_B12 = (gn_B12 - fn_B12 * dot_partial) * phase2_inv_norm;

                        const float fn_B12_prev = my_features[3 * half_D + prev_idx];
                        const float g_B12_prev = (gn_B12_prev - fn_B12_prev * dot_partial) * phase2_inv_norm;

                        const float x0 = my_x0[p];
                        const float x1 = my_x1[p];
                        const int next = (p + 1) & (half_D - 1);
                        const float x0n = my_x0[next];
                        const float x1n = my_x1[next];
                        const float x0p = my_x0[prev_idx];
                        const float x1p = my_x1[prev_idx];

                        float g_x0 = g_S0 * 2.0f * x0 + g_V1 + g_B12 * x1n + g_B12_prev * (-x1p) + my_gx0_rot[p];
                        float g_x1 = g_S0 * 2.0f * x1 + g_V2 + g_B12 * (-x0n) + g_B12_prev * x0p + my_gx1_rot[p];

                        my_work[shmem_idx(p0_perm)] = g_x0;
                        my_work[shmem_idx(p1_perm)] = g_x1;
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (token_slot == 0) {
                        const int a_base = ((b * K + k) * H + hh) * half_D;
                        const int s_base_off = ((b * K + k) * H + hh) * D;

                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            ga_global(grad_angles[a_base + p]);
                        ga_global.fetch_add(grad_angles_accum[p]);

                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            gs0_global(grad_scales[s_base_off + 2 * p]);
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            gs1_global(grad_scales[s_base_off + 2 * p + 1]);
                        gs0_global.fetch_add(grad_scales_accum[2 * p]);
                        gs1_global.fetch_add(grad_scales_accum[2 * p + 1]);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (valid) {
                    if (B == 1) {
                        grad_x[x_offset + 2 * p]     = my_work[shmem_idx(2 * p)];
                        grad_x[x_offset + 2 * p + 1] = my_work[shmem_idx(2 * p + 1)];
                    } else {
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            gx0_ref(grad_x[x_offset + 2 * p]);
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            gx1_ref(grad_x[x_offset + 2 * p + 1]);
                        gx0_ref.fetch_add(my_work[shmem_idx(2 * p)]);
                        gx1_ref.fetch_add(my_work[shmem_idx(2 * p + 1)]);
                    }
                }
            });
    });
}

extern "C" void clifford_fwd_launcher(
    const float* x_pairs,
    float* features,
    int batch,
    int P)
{
    if (batch <= 0 || P <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = std::max(1, std::min(P, 256));
    sycl::range<1> global_size(static_cast<size_t>(batch) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class k_clifford_fwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int batch_idx = item.get_group(0);
                if (batch_idx >= batch) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);

                for (int pair_idx = tid; pair_idx < P; pair_idx += block_dim) {
                    int base_in = (batch_idx * P + pair_idx) * 2;
                    float x0 = x_pairs[base_in];
                    float x1 = x_pairs[base_in + 1];

                    int next_idx = (pair_idx + 1) % P;
                    int next_in = (batch_idx * P + next_idx) * 2;
                    float x0_next = x_pairs[next_in];
                    float x1_next = x_pairs[next_in + 1];

                    int base_out = batch_idx * 4 * P;
                    features[base_out + pair_idx]         = x0 * x0 + x1 * x1;
                    features[base_out + P + pair_idx]     = x0;
                    features[base_out + 2 * P + pair_idx] = x1;
                    features[base_out + 3 * P + pair_idx] = x0 * x1_next - x1 * x0_next;
                }
            });
    });
}

extern "C" void clifford_bwd_launcher(
    const float* x_pairs,
    const float* grad_features,
    float* grad_x_pairs,
    int batch,
    int P)
{
    if (batch <= 0 || P <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = std::max(1, std::min(P, 256));
    sycl::range<1> global_size(static_cast<size_t>(batch) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class k_clifford_bwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int batch_idx = item.get_group(0);
                if (batch_idx >= batch) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);

                for (int pair_idx = tid; pair_idx < P; pair_idx += block_dim) {
                    int base_in = (batch_idx * P + pair_idx) * 2;
                    int base_grad = batch_idx * 4 * P;

                    float x0 = x_pairs[base_in];
                    float x1 = x_pairs[base_in + 1];

                    float gs   = grad_features[base_grad + pair_idx];
                    float gv1  = grad_features[base_grad + P + pair_idx];
                    float gv2  = grad_features[base_grad + 2 * P + pair_idx];
                    float gb12 = grad_features[base_grad + 3 * P + pair_idx];

                    int next_idx = (pair_idx + 1) % P;
                    float x0_next = x_pairs[(batch_idx * P + next_idx) * 2];
                    float x1_next = x_pairs[(batch_idx * P + next_idx) * 2 + 1];

                    int prev_idx = (pair_idx - 1 + P) % P;
                    float x0_prev   = x_pairs[(batch_idx * P + prev_idx) * 2];
                    float x1_prev   = x_pairs[(batch_idx * P + prev_idx) * 2 + 1];
                    float gb12_prev = grad_features[base_grad + 3 * P + prev_idx];

                    float gx0 = 2.0f * x0 * gs + gv1 + gb12 * x1_next + gb12_prev * (-x1_prev);
                    float gx1 = 2.0f * x1 * gs + gv2 + gb12 * (-x0_next) + gb12_prev * x0_prev;

                    grad_x_pairs[base_in]     = gx0;
                    grad_x_pairs[base_in + 1] = gx1;
                }
            });
    });
}

extern "C" void fused_embedding_fwd_launcher(
    const float* phase_angles,
    const float* log_magnitude,
    const int* token_ids,
    float* output,
    int N, int P)
{
    if (N <= 0 || P <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = std::max(1, std::min(P, 256));
    sycl::range<1> global_size(static_cast<size_t>(N) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class k_fused_embedding_fwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int n = item.get_group(0);
                if (n >= N) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);

                int token_id = token_ids[n];
                float log_mag = sycl::fmin(sycl::fmax(log_magnitude[token_id], -5.0f), 5.0f);
                float mag = sycl::exp(log_mag);

                for (int p = tid; p < P; p += block_dim) {
                    float angle = phase_angles[token_id * P + p];
                    float c = sycl::cos(angle);
                    float s = sycl::sin(angle);
                    output[n * 2 * P + 2 * p]     = c * mag;
                    output[n * 2 * P + 2 * p + 1] = s * mag;
                }
            });
    });
}

extern "C" void fused_embedding_bwd_launcher(
    const float* phase_angles,
    const float* log_magnitude,
    const int* token_ids,
    const float* grad_output,
    float* grad_phase_angles,
    float* grad_log_magnitude,
    int N, int P)
{
    if (N <= 0 || P <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = std::max(1, std::min(P, 256));
    sycl::range<1> global_size(static_cast<size_t>(N) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> s_lm_sum(sycl::range<1>(1), h);

        h.parallel_for<class k_fused_embedding_bwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int n = item.get_group(0);
                if (n >= N) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);

                if (tid == 0) s_lm_sum[0] = 0.0f;
                item.barrier(sycl::access::fence_space::local_space);

                int token_id = token_ids[n];
                float raw_log_mag = log_magnitude[token_id];
                float log_mag = sycl::fmin(sycl::fmax(raw_log_mag, -5.0f), 5.0f);
                float mag = sycl::exp(log_mag);

                float local_lm_grad = 0.0f;
                for (int p = tid; p < P; p += block_dim) {
                    float angle = phase_angles[token_id * P + p];
                    float c = sycl::cos(angle);
                    float s = sycl::sin(angle);

                    float g0 = grad_output[n * 2 * P + 2 * p];
                    float g1 = grad_output[n * 2 * P + 2 * p + 1];

                    float grad_angle = g0 * (-s * mag) + g1 * (c * mag);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        gpa(grad_phase_angles[token_id * P + p]);
                    gpa.fetch_add(grad_angle);

                    if (raw_log_mag >= -5.0f && raw_log_mag <= 5.0f) {
                        local_lm_grad += g0 * (c * mag) + g1 * (s * mag);
                    }
                }

                if (local_lm_grad != 0.0f) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        slm(s_lm_sum[0]);
                    slm.fetch_add(local_lm_grad);
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (tid == 0 && s_lm_sum[0] != 0.0f) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        glm(grad_log_magnitude[token_id]);
                    glm.fetch_add(s_lm_sum[0]);
                }
            });
    });
}

extern "C" void fused_norm_fwd_launcher(
    const float* x, const float* scale, float* out, float* rms_inv_out,
    int N, int D, float eps)
{
    if (N <= 0 || D <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int half_D = D / 2;
    int threads = 1;
    while (threads * 2 <= std::min(half_D, 256)) threads *= 2;

    sycl::range<1> global_size(static_cast<size_t>(N) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(threads), h);

        h.parallel_for<class k_fused_norm_fwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int n = item.get_group(0);
                if (n >= N) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);

                float local_sum = 0.0f;
                for (int p = tid; p < half_D; p += block_dim) {
                    float x0 = x[n * D + 2 * p];
                    float x1 = x[n * D + 2 * p + 1];
                    local_sum += x0 * x0 + x1 * x1;
                }
                sdata[tid] = local_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int s = block_dim / 2; s > 0; s >>= 1) {
                    if (tid < s) sdata[tid] += sdata[tid + s];
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float rms_inv = sycl::rsqrt(sdata[0] / half_D + eps);
                if (tid == 0) {
                    rms_inv_out[n] = rms_inv;
                }

                for (int i = tid; i < D; i += block_dim) {
                    out[n * D + i] = x[n * D + i] * rms_inv * scale[i];
                }
            });
    });
}

extern "C" void fused_norm_bwd_launcher(
    const float* grad_out, const float* x, const float* scale,
    const float* rms_inv, float* grad_x,
    int N, int D, float eps)
{
    if (N <= 0 || D <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int half_D = D / 2;
    int threads = 1;
    while (threads * 2 <= std::min(D, 256)) threads *= 2;

    sycl::range<1> global_size(static_cast<size_t>(N) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(threads), h);

        h.parallel_for<class k_fused_norm_bwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int n = item.get_group(0);
                if (n >= N) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);

                float rms_inv_n = rms_inv[n];
                float local_dot = 0.0f;
                for (int i = tid; i < D; i += block_dim) {
                    local_dot += grad_out[n * D + i] * scale[i] * x[n * D + i];
                }
                sdata[tid] = local_dot;
                item.barrier(sycl::access::fence_space::local_space);

                for (int s = block_dim / 2; s > 0; s >>= 1) {
                    if (tid < s) sdata[tid] += sdata[tid + s];
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float coeff = sdata[0] * rms_inv_n * rms_inv_n / half_D;
                for (int i = tid; i < D; i += block_dim) {
                    float go = grad_out[n * D + i];
                    float xi = x[n * D + i];
                    float si = scale[i];
                    grad_x[n * D + i] = (go * si - xi * coeff) * rms_inv_n;
                }
            });
    });
}

extern "C" void fused_gelu_fwd_launcher(
    const float* x, float* out,
    int total_pairs, int half_D, float eps)
{
    if (total_pairs <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;

    sycl::range<1> global_size(static_cast<size_t>(blocks) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class k_fused_gelu_fwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int idx = item.get_global_linear_id();
                if (idx >= total_pairs) return;

                int n = idx / half_D;
                int p = idx % half_D;
                int offset = n * 2 * half_D + 2 * p;

                float x0 = x[offset];
                float x1 = x[offset + 1];
                float r = sycl::sqrt(x0 * x0 + x1 * x1);
                float gelu_r = 0.5f * r * (1.0f + sycl::erf(r * 0.7071067811865476f));
                float gate = gelu_r / (r + eps);

                out[offset]     = x0 * gate;
                out[offset + 1] = x1 * gate;
            });
    });
}

extern "C" void fused_gelu_bwd_launcher(
    const float* grad_out, const float* x, float* grad_x,
    int total_pairs, int half_D, float eps)
{
    if (total_pairs <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;

    sycl::range<1> global_size(static_cast<size_t>(blocks) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class k_fused_gelu_bwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int idx = item.get_global_linear_id();
                if (idx >= total_pairs) return;

                int n = idx / half_D;
                int p = idx % half_D;
                int offset = n * 2 * half_D + 2 * p;

                float x0 = x[offset];
                float x1 = x[offset + 1];
                float r = sycl::sqrt(x0 * x0 + x1 * x1);
                float r_eps = r + eps;

                float erf_val = sycl::erf(r * 0.7071067811865476f);
                float gelu_r = 0.5f * r * (1.0f + erf_val);
                float gate = gelu_r / r_eps;

                float exp_val = sycl::exp(-0.5f * r * r);
                float gelu_prime = 0.5f * (1.0f + erf_val) + r * exp_val * 0.3989422804014327f;
                float dgate_dr = (gelu_prime - gate) / r_eps;

                float g0 = grad_out[offset];
                float g1 = grad_out[offset + 1];

                if (r > 1e-12f) {
                    float inv_r = 1.0f / r;
                    float common = dgate_dr * inv_r * (x0 * g0 + x1 * g1);
                    grad_x[offset]     = gate * g0 + x0 * common;
                    grad_x[offset + 1] = gate * g1 + x1 * common;
                } else {
                    grad_x[offset]     = gate * g0;
                    grad_x[offset + 1] = gate * g1;
                }
            });
    });
}

extern "C" void fused_mixer_fwd_launcher(
    const float* x_in,
    const float* angles, const float* coupling, const float* raw_scales,
    float* x_out,
    float* save_feat_norm, float* save_inv_norm, float* save_theta,
    int N, int D, int n_groups, int G, float eps)
{
    if (N <= 0 || D <= 0 || n_groups <= 0 || G <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();

    int TPB = std::max(1, std::min(MIXER_TPB, 256 / std::max(1, G)));
    int n_tiles = (N + TPB - 1) / TPB;
    int n_blocks = n_tiles * n_groups;
    int F_FEAT = 4;
    int coup_tile_size = F_FEAT * G * G;
    int smem_total = coup_tile_size + 2 * TPB * G + TPB * F_FEAT * G + 2 * G + TPB;

    sycl::range<2> global_size(static_cast<size_t>(n_blocks) * TPB, G);
    sycl::range<2> local_size(TPB, G);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_total), h);

        h.parallel_for<class k_fused_mixer_fwd>(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) {
                int t = item.get_local_id(0);
                int p = item.get_local_id(1);
                int block_id = item.get_group(0);

                int token_tile = block_id / n_groups;
                int g = block_id % n_groups;
                int n = token_tile * TPB + t;
                bool valid = (n < N);

                float* s_ptr = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* s_coup = s_ptr;
                float* s_x0   = s_coup + coup_tile_size;
                float* s_x1   = s_x0 + TPB * G;
                float* s_feat = s_x1 + TPB * G;
                float* s_scl  = s_feat + TPB * F_FEAT * G;
                float* s_inv_norm = s_scl + 2 * G;

                const float* coup_g = coupling + g * coup_tile_size;
                for (int i = p + t * G; i < coup_tile_size; i += G * TPB) {
                    s_coup[i] = coup_g[i];
                }

                if (t == 0) {
                    s_scl[p]     = softplus_stable_sycl(raw_scales[g * 2 * G + 2 * p]);
                    s_scl[p + G] = softplus_stable_sycl(raw_scales[g * 2 * G + 2 * p + 1]);
                }

                if (valid) {
                    int pair_idx = g * G + p;
                    int d_base = 2 * pair_idx;
                    s_x0[t * G + p] = x_in[n * D + d_base];
                    s_x1[t * G + p] = x_in[n * D + d_base + 1];
                } else {
                    s_x0[t * G + p] = 0.0f;
                    s_x1[t * G + p] = 0.0f;
                }
                item.barrier(sycl::access::fence_space::local_space);

                float x0 = s_x0[t * G + p];
                float x1 = s_x1[t * G + p];
                int p_next = (p + 1) % G;
                float x0_next = s_x0[t * G + p_next];
                float x1_next = s_x1[t * G + p_next];

                s_feat[t * F_FEAT * G + 0 * G + p] = x0 * x0 + x1 * x1;
                s_feat[t * F_FEAT * G + 1 * G + p] = x0;
                s_feat[t * F_FEAT * G + 2 * G + p] = x1;
                s_feat[t * F_FEAT * G + 3 * G + p] = x0 * x0_next - x1 * x1_next;
                item.barrier(sycl::access::fence_space::local_space);

                if (p == 0) {
                    float sq = 0.0f;
                    for (int j = 0; j < F_FEAT * G; j++) {
                        float v = s_feat[t * F_FEAT * G + j];
                        sq += v * v;
                    }
                    float inv = 1.0f / (sycl::sqrt(sq) + eps);
                    s_inv_norm[t] = inv;
                    for (int j = 0; j < F_FEAT * G; j++) {
                        s_feat[t * F_FEAT * G + j] *= inv;
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                float mod = 0.0f;
                for (int j = 0; j < F_FEAT * G; j++) {
                    mod += s_feat[t * F_FEAT * G + j] * s_coup[j * G + p];
                }

                float theta = angles[g * G + p] + sycl::tanh(mod);
                float c = sycl::cos(theta);
                float s = sycl::sin(theta);

                float r0 = x0 * c - x1 * s;
                float r1 = x0 * s + x1 * c;

                float scl0 = s_scl[p];
                float scl1 = s_scl[p + G];

                if (valid) {
                    int pair_idx = g * G + p;
                    int d_base = 2 * pair_idx;
                    x_out[n * D + d_base]     = r0 * scl0;
                    x_out[n * D + d_base + 1] = r1 * scl1;
                }

                if (save_feat_norm != nullptr && valid) {
                    for (int j = p; j < F_FEAT * G; j += G) {
                        save_feat_norm[n * n_groups * F_FEAT * G + g * F_FEAT * G + j] =
                            s_feat[t * F_FEAT * G + j];
                    }
                }
                if (save_inv_norm != nullptr && valid && p == 0) {
                    save_inv_norm[n * n_groups + g] = s_inv_norm[t];
                }
                if (save_theta != nullptr && valid) {
                    save_theta[n * n_groups * G + g * G + p] = theta;
                }
            });
    });
}

extern "C" void fused_mixer_bwd_launcher(
    const float* grad_out, const float* x_in,
    const float* angles, const float* coupling, const float* raw_scales,
    const float* save_feat_norm, const float* save_inv_norm, const float* save_theta,
    float* grad_x, float* grad_angles, float* grad_coupling, float* grad_raw_scales,
    int N, int D, int n_groups, int G, float eps)
{
    if (N <= 0 || D <= 0 || n_groups <= 0 || G <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();

    int TPB = std::max(1, std::min(MIXER_TPB, 256 / std::max(1, G)));
    int n_tiles = (N + TPB - 1) / TPB;
    int n_blocks = n_tiles * n_groups;
    int F_FEAT = 4;
    int coup_tile_size = F_FEAT * G * G;
    int smem_total = coup_tile_size
                   + TPB * F_FEAT * G
                   + 3 * TPB * G
                   + TPB * F_FEAT * G
                   + G + 2 * G + coup_tile_size
                   + 2 * TPB;

    sycl::range<2> global_size(static_cast<size_t>(n_blocks) * TPB, G);
    sycl::range<2> local_size(TPB, G);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>(smem_total), h);

        h.parallel_for<class k_fused_mixer_bwd>(
            sycl::nd_range<2>(global_size, local_size),
            [=](sycl::nd_item<2> item) {
                int t = item.get_local_id(0);
                int p = item.get_local_id(1);
                int block_id = item.get_group(0);

                int token_tile = block_id / n_groups;
                int g = block_id % n_groups;
                int n = token_tile * TPB + t;
                bool valid = (n < N);

                float* s_ptr      = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                float* s_coup     = s_ptr;
                float* s_featn    = s_coup  + coup_tile_size;
                float* s_x0       = s_featn + TPB * F_FEAT * G;
                float* s_x1       = s_x0    + TPB * G;
                float* s_dmod     = s_x1    + TPB * G;
                float* s_dfeat    = s_dmod  + TPB * G;
                float* s_gang     = s_dfeat + TPB * F_FEAT * G;
                float* s_gscl     = s_gang  + G;
                float* s_gcoup    = s_gscl  + 2 * G;
                float* s_inv_norm = s_gcoup + coup_tile_size;
                float* s_a        = s_inv_norm + TPB;

                const float* coup_g = coupling + g * coup_tile_size;
                for (int i = p + t * G; i < coup_tile_size; i += G * TPB) {
                    s_coup[i] = coup_g[i];
                }
                for (int i = p + t * G; i < G; i += G * TPB) s_gang[i] = 0.0f;
                for (int i = p + t * G; i < 2 * G; i += G * TPB) s_gscl[i] = 0.0f;
                for (int i = p + t * G; i < coup_tile_size; i += G * TPB) s_gcoup[i] = 0.0f;

                float x0 = 0.0f, x1 = 0.0f, theta = 0.0f;
                if (valid) {
                    int pair_idx = g * G + p;
                    int d_base = 2 * pair_idx;
                    x0 = x_in[n * D + d_base];
                    x1 = x_in[n * D + d_base + 1];
                    s_x0[t * G + p] = x0;
                    s_x1[t * G + p] = x1;
                    theta = save_theta[n * n_groups * G + g * G + p];
                    for (int j = p; j < F_FEAT * G; j += G) {
                        s_featn[t * F_FEAT * G + j] =
                            save_feat_norm[n * n_groups * F_FEAT * G + g * F_FEAT * G + j];
                    }
                } else {
                    s_x0[t * G + p] = 0.0f;
                    s_x1[t * G + p] = 0.0f;
                    for (int j = p; j < F_FEAT * G; j += G) {
                        s_featn[t * F_FEAT * G + j] = 0.0f;
                    }
                }
                if (p == 0) {
                    s_inv_norm[t] = valid ? save_inv_norm[n * n_groups + g] : 0.0f;
                }

                float raw_scl0 = raw_scales[g * 2 * G + 2 * p];
                float raw_scl1 = raw_scales[g * 2 * G + 2 * p + 1];
                float scl0 = softplus_stable_sycl(raw_scl0);
                float scl1 = softplus_stable_sycl(raw_scl1);
                float sig0 = 1.0f / (1.0f + sycl::exp(-raw_scl0));
                float sig1 = 1.0f / (1.0f + sycl::exp(-raw_scl1));

                item.barrier(sycl::access::fence_space::local_space);

                float dout0 = 0.0f, dout1 = 0.0f;
                if (valid) {
                    int pair_idx = g * G + p;
                    int d_base = 2 * pair_idx;
                    dout0 = grad_out[n * D + d_base];
                    dout1 = grad_out[n * D + d_base + 1];
                }

                float dr0 = dout0 * scl0;
                float dr1 = dout1 * scl1;

                float c_angle = sycl::cos(theta);
                float s_angle = sycl::sin(theta);
                float r0 = x0 * c_angle - x1 * s_angle;
                float r1 = x0 * s_angle + x1 * c_angle;

                {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        gs0(s_gscl[2 * p]);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        gs1(s_gscl[2 * p + 1]);
                    gs0.fetch_add(dout0 * r0 * sig0);
                    gs1.fetch_add(dout1 * r1 * sig1);
                }

                float dtheta = -r1 * dr0 + r0 * dr1;
                {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>
                        ga(s_gang[p]);
                    ga.fetch_add(dtheta);
                }

                float dx0_rot = dr0 * c_angle + dr1 * s_angle;
                float dx1_rot = -dr0 * s_angle + dr1 * c_angle;

                float tanh_mod = theta - angles[g * G + p];
                float dmod = dtheta * (1.0f - tanh_mod * tanh_mod);
                s_dmod[t * G + p] = dmod;

                item.barrier(sycl::access::fence_space::local_space);

                {
                    float dmod_p = s_dmod[t * G + p];
                    for (int j = 0; j < F_FEAT * G; j++) {
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::work_group,
                                         sycl::access::address_space::local_space>
                            gc(s_gcoup[j * G + p]);
                        gc.fetch_add(s_featn[t * F_FEAT * G + j] * dmod_p);
                    }
                }

                for (int jp = 0; jp < F_FEAT; jp++) {
                    int j = jp * G + p;
                    float df = 0.0f;
                    for (int pp = 0; pp < G; pp++) {
                        df += s_dmod[t * G + pp] * s_coup[j * G + pp];
                    }
                    s_dfeat[t * F_FEAT * G + j] = df;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (p == 0) {
                    float a_val = 0.0f;
                    for (int j = 0; j < F_FEAT * G; j++) {
                        a_val += s_dfeat[t * F_FEAT * G + j] * s_featn[t * F_FEAT * G + j];
                    }
                    s_a[t] = a_val;
                }
                item.barrier(sycl::access::fence_space::local_space);

                {
                    float inv = s_inv_norm[t];
                    float denom = 1.0f - eps * inv;
                    float safe_inv_denom = (sycl::fabs(denom) > 1e-8f) ? (1.0f / denom) : 0.0f;
                    float a_val = s_a[t];
                    for (int jp = 0; jp < F_FEAT; jp++) {
                        int j = jp * G + p;
                        float dfn = s_dfeat[t * F_FEAT * G + j];
                        float fn  = s_featn[t * F_FEAT * G + j];
                        s_dfeat[t * F_FEAT * G + j] = inv * (dfn - fn * a_val * safe_inv_denom);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);

                int p_next = (p + 1) % G;
                int p_prev = (p - 1 + G) % G;

                float d_S0       = s_dfeat[t * F_FEAT * G + 0 * G + p];
                float d_V1       = s_dfeat[t * F_FEAT * G + 1 * G + p];
                float d_V2       = s_dfeat[t * F_FEAT * G + 2 * G + p];
                float d_B12_p    = s_dfeat[t * F_FEAT * G + 3 * G + p];
                float d_B12_prev = s_dfeat[t * F_FEAT * G + 3 * G + p_prev];

                float x0_next = s_x0[t * G + p_next];
                float x1_next = s_x1[t * G + p_next];
                float x0_prev = s_x0[t * G + p_prev];
                float x1_prev = s_x1[t * G + p_prev];

                float dx0_feat = d_S0 * 2.0f * x0 + d_V1 + d_B12_p * x0_next + d_B12_prev * x0_prev;
                float dx1_feat = d_S0 * 2.0f * x1 + d_V2 - d_B12_p * x1_next - d_B12_prev * x1_prev;

                if (valid) {
                    int pair_idx = g * G + p;
                    int d_base = 2 * pair_idx;
                    grad_x[n * D + d_base]     = dx0_rot + dx0_feat;
                    grad_x[n * D + d_base + 1] = dx1_rot + dx1_feat;
                }
                item.barrier(sycl::access::fence_space::local_space);

                if (t == 0) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        ga_glob(grad_angles[g * G + p]);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        gs0_glob(grad_raw_scales[g * 2 * G + 2 * p]);
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        gs1_glob(grad_raw_scales[g * 2 * G + 2 * p + 1]);
                    ga_glob.fetch_add(s_gang[p]);
                    gs0_glob.fetch_add(s_gscl[2 * p]);
                    gs1_glob.fetch_add(s_gscl[2 * p + 1]);
                }
                for (int i = p + t * G; i < coup_tile_size; i += G * TPB) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        gc_glob(grad_coupling[g * coup_tile_size + i]);
                    gc_glob.fetch_add(s_gcoup[i]);
                }
            });
    });
}

extern "C" void fused_zloss_fwd_launcher(
    const float* logits, const int* mask, float* lse_out,
    int N, int V)
{
    if (N <= 0 || V <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = 256;
    sycl::range<1> global_size(static_cast<size_t>(N) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(threads), h);

        h.parallel_for<class k_fused_zloss_fwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int n = item.get_group(0);
                if (n >= N) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);
                const float* row = logits + static_cast<long long>(n) * V;

                if (mask[n] == 0) {
                    if (tid == 0) lse_out[n] = 0.0f;
                    return;
                }

                float local_max = -INFINITY;
                for (int v = tid; v < V; v += block_dim) {
                    local_max = sycl::fmax(local_max, row[v]);
                }
                sdata[tid] = local_max;
                item.barrier(sycl::access::fence_space::local_space);
                for (int s = block_dim / 2; s > 0; s >>= 1) {
                    if (tid < s) sdata[tid] = sycl::fmax(sdata[tid], sdata[tid + s]);
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float row_max = sdata[0];
                item.barrier(sycl::access::fence_space::local_space);

                float local_sum = 0.0f;
                for (int v = tid; v < V; v += block_dim) {
                    local_sum += sycl::exp(row[v] - row_max);
                }
                sdata[tid] = local_sum;
                item.barrier(sycl::access::fence_space::local_space);
                for (int s = block_dim / 2; s > 0; s >>= 1) {
                    if (tid < s) sdata[tid] += sdata[tid + s];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (tid == 0) {
                    lse_out[n] = sycl::log(sdata[0]) + row_max;
                }
            });
    });
}

extern "C" void fused_zloss_bwd_launcher(
    const float* logits, const float* lse, const int* mask,
    float grad_scale, float* grad_logits,
    int N, int V)
{
    if (N <= 0 || V <= 0) return;
    sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
    int threads = 256;
    sycl::range<1> global_size(static_cast<size_t>(N) * threads);
    sycl::range<1> local_size(threads);

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class k_fused_zloss_bwd>(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int n = item.get_group(0);
                if (n >= N) return;
                int tid = item.get_local_id(0);
                int block_dim = item.get_local_range(0);
                const float* row = logits + static_cast<long long>(n) * V;
                float* grow = grad_logits + static_cast<long long>(n) * V;

                if (mask[n] == 0) {
                    for (int v = tid; v < V; v += block_dim) grow[v] = 0.0f;
                    return;
                }

                float lse_n = lse[n];
                float coef = grad_scale * lse_n;
                for (int v = tid; v < V; v += block_dim) {
                    grow[v] = coef * sycl::exp(row[v] - lse_n);
                }
            });
    });
}

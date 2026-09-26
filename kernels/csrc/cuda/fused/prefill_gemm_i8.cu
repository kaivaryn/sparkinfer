// int8 tensor-core GEMM for Qwythos batched prefill (see prefill_i8.h).
//
// C[M,N] = A[M,K] @ W^T, W native GGUF [N,K] row-major. int8 x int8 -> int32 with the dequant
// (per-token sx[m] * per-channel sw[n]) folded into the store, emitting bf16 C.
//
// Shaped for int8 rather than mirroring the bf16 GEMM: mma.sync m16n8k32 (int8's native shape --
// wmma m16n16k16 can only emit the k16 shape, which caps at half the int8 MAC rate), BK=64 to halve
// the main-loop barrier count, an XOR-swizzled smem layout so the 4B operand loads spread across
// banks, and a register->global epilogue that keeps the int32 accumulators out of shared memory.
// Same int8 quantization scheme and accumulation order as before, so C is bit-identical.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include "sparkinfer/kernels/prefill_i8.h"

#include "sparkinfer/kernels/prefill_quant_rows.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int PF_BM = 128;
constexpr int PF_BN = 128;
constexpr int PF_BK = 64;          // 4 x 16B chunks per row
constexpr int PF_MFRAG = 2;        // 32 rows per warp / 16
constexpr int PF_NFRAG = 8;        // 64 cols per warp / 8

__device__ __forceinline__ void pf_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// XOR swizzle at 16B granularity. A 64B row covers 16 banks, so rows r and r+1 share one 128B bank
// line and an 8-row ldmatrix phase needs rows {0,2,4,6} (and {1,3,5,7}) on four DIFFERENT chunks.
// `c ^ (r & 3)` -- laid out for the 4B lds.32 walk this kernel no longer does -- puts rows 0 and 4
// on the same chunk, so every operand ldmatrix pays a 2-way conflict. `c ^ ((r >> 1) & 3)` gives the
// eight rows eight distinct 16B bank groups. Stores and loads both go through it, so smem holds
// the same bytes at different addresses and C is unchanged. SWZ=0 keeps the old map
// (SPARKINFER_PREFILL_GEMM_I8_SWZ=0).
template <int SWZ = 1>
__device__ __forceinline__ int pf_swz(int k, int row) {
    if constexpr (SWZ == 0) return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
    return (((k >> 4) ^ ((row >> 1) & 3)) << 4) | (k & 15);
}

// ldmatrix.x4: one instruction moves all four 8x8 operand tiles of an mma fragment through the
// LDS pipe (the four scalar lds.32 it replaces issued 1:1 against the mma pipe and were the
// staging bottleneck). Thread t supplies the shared-memory address of row (t&7) of tile (t>>3);
// the XOR swizzle still applies per row address, so the smem layout is unchanged and the loaded
// registers are identical to the lds.32 path.
__device__ __forceinline__ void pf_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                          const signed char* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Per-row symmetric int8 quantize, one warp per row.
__global__ void pf_quantize_rows_i8(const __nv_bfloat16* __restrict__ x, signed char* __restrict__ q,
                                    float* __restrict__ scale, int rows, int cols) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    float amax = 0.f;
    for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(__bfloat162float(x[(size_t)r * cols + c])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) scale[r] = d;
    for (int c = lane; c < cols; c += 32)
        q[(size_t)r * cols + c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(__bfloat162float(x[(size_t)r * cols + c]) / d));
}

// The 2 in __launch_bounds__ is required, not decorative: left to itself nvcc picks 131 registers,
// and 2 * 256 * 131 exceeds the 65536-register file, so only one block per SM would be resident.
// RESID folds the residual add into the store: C[m,n] = bf16(C[m,n] + bf16(acc*sx*sw)) -- the same
// two-step rounding as the pf_add kernel it replaces, reading and writing through the ONE C
// pointer (no second aliased argument), so the fused path stays bit-identical to GEMM-then-add.
//
// SPLITK partitions the K loop across blockIdx.z (ktiles BK-tiles each) and accumulates the int32
// tile into P[M,N] with atomicAdd instead of storing C; a separate epilogue applies sx/sw. The
// output is bit-identical because the accumulator is int32: integer addition is exact and
// associative, so the reordered partial sums land on the same value the single-block loop produces.
// FULL: M, N and K are exact multiples of the tile, so every stage and every epilogue store is
// in range. The predicates are data-dependent otherwise and ptxas cannot delete them; on the
// aligned FFN shapes they are pure issue overhead around the copies and the store. Same addresses,
// same values. SPARKINFER_PREFILL_GEMM_I8_ALIGNED=0 keeps the checked kernel.
template <bool RESID, bool SPLITK, bool FULL = false, int SWZ = 1>
__global__ __launch_bounds__(256, 2) void pf_gemm_i8_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* C, int* __restrict__ P, int M, int N, int K, int ktiles) {
    __shared__ signed char As[2][PF_BM][PF_BK];
    __shared__ signed char Bs[2][PF_BN][PF_BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7
    const int tig  = lane & 3;                        // thread-in-group
    const int sub  = lane >> 3;                       // ldmatrix tile this thread addresses (0..3)
    const int lrow = lane & 7;                        // row within that tile
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int m0   = blockIdx.y * PF_BM;
    const int n0   = blockIdx.x * PF_BN;
    const int nk   = (K + PF_BK - 1) / PF_BK;
    // K-tile range this block owns. Without SPLITK that is the whole K loop, exactly as before.
    int t0 = 0, t1 = nk;
    if (SPLITK) {
        t0 = blockIdx.z * ktiles;
        t1 = t0 + ktiles;
        if (t1 > nk) t1 = nk;
        if (t0 >= t1) return;
    }

    int acc[PF_MFRAG][PF_NFRAG][4];
    #pragma unroll
    for (int i = 0; i < PF_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < PF_NFRAG; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

    // 128 rows x 64B = 512 16B chunks per tile; 256 threads stage 2 A-chunks + 2 B-chunks each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gm = m0 + r, gn = n0 + r, gk = k0 + k;
            if constexpr (FULL) {
                __pipeline_memcpy_async(&As[buf][r][pf_swz<SWZ>(k, r)], &A[(size_t)gm * K + gk], 16);
                __pipeline_memcpy_async(&Bs[buf][r][pf_swz<SWZ>(k, r)], &W[(size_t)gn * K + gk], 16);
            } else {
                pf_cp16(&As[buf][r][pf_swz<SWZ>(k, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
                pf_cp16(&Bs[buf][r][pf_swz<SWZ>(k, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
            }
        }
        __pipeline_commit();
    };

    stage(0, t0 * PF_BK);
    int buf = 0;
    for (int t = t0; t < t1; t++) {
        if (t + 1 < t1) stage(buf ^ 1, (t + 1) * PF_BK);
        __pipeline_wait_prior(t + 1 < t1 ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 32) {
            unsigned af[PF_MFRAG][4], bf[PF_NFRAG][2];
            // A fragment i: tiles {rows lo,k0} {rows hi,k0} {rows lo,k16} {rows hi,k16} -> af[i][0..3]
            #pragma unroll
            for (int i = 0; i < PF_MFRAG; i++) {
                const int row = wm * 32 + i * 16 + (sub & 1) * 8 + lrow;
                pf_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                          &As[buf][row][pf_swz<SWZ>(kk + (sub >> 1) * 16, row)]);
            }
            // B pair (j, j+1): tiles {cols j,k0} {cols j,k16} {cols j+1,k0} {cols j+1,k16}
            #pragma unroll
            for (int jp = 0; jp < PF_NFRAG; jp += 2) {
                const int col = wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                pf_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                          &Bs[buf][col][pf_swz<SWZ>(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < PF_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < PF_NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        __syncthreads();
        buf ^= 1;
    }

    // Split-K: hand the int32 tile to the partial buffer and let the epilogue scale it. Same
    // (row, col) map as the scalar tail below.
    if (SPLITK) {
        #pragma unroll
        for (int i = 0; i < PF_MFRAG; i++) {
            #pragma unroll
            for (int j = 0; j < PF_NFRAG; j++) {
                const int gn = n0 + wn * 64 + j * 8 + tig * 2;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N) atomicAdd(&P[(size_t)gm * N + cn], acc[i][j][e]);
                }
            }
        }
        return;
    }

    // Registers straight to global: c0/c1 (and c2/c3) are adjacent columns, so each pair packs into
    // one 4B bf16x2 store. Dead on the split-K instantiation (it returned above).
    if constexpr (!SPLITK)
    #pragma unroll
    for (int i = 0; i < PF_MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < PF_NFRAG; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            if constexpr (!FULL)
            if (gn + 1 >= N) {                        // tail: scalar path
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N) {
                        __nv_bfloat16 v = __float2bfloat16((float)acc[i][j][e] * sx[gm] * sw[cn]);
                        if (RESID)
                            v = __float2bfloat16(__bfloat162float(C[(size_t)gm * N + cn]) +
                                                 __bfloat162float(v));
                        C[(size_t)gm * N + cn] = v;
                    }
                }
                continue;
            }
            const float w0 = sw[gn], w1 = sw[gn + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                if constexpr (!FULL)
                    if (gm >= M) continue;
                const float s = sx[gm];
                __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                         (float)acc[i][j][h * 2 + 1] * s * w1);
                __nv_bfloat162* cp = reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]);
                if (RESID) {
                    const __nv_bfloat162 r = *cp;
                    v = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(v.x),
                                              __bfloat162float(r.y) + __bfloat162float(v.y));
                }
                *cp = v;
            }
        }
    }
}
// FULL-tile kernel with a 64x64 warp tile: the same 128x128 block, four warps instead of eight.
// Each warp's k32 step issues 8 ldmatrix.x4 for 32 mma (the 32x64 warp tile above: 6 for 16), a
// third less shared-memory traffic per MAC, and the half-size block buys a third cp.async stage in
// the same 48 KB of smem while two blocks still fit an SM. The accumulator is int32 and the
// epilogue is the per-element arithmetic of pf_gemm_i8_kernel, so C is bit-identical.
// SPARKINFER_PREFILL_GEMM_I8_W4=0 keeps pf_gemm_i8_kernel.
//
// GRP > 0 walks the output on a 1-D grid, GRP M-tiles at a time across every N-tile, instead of
// N-fastest over the whole M range, so the ~340 live tiles share fewer weight columns. Measured in
// the model on Ternary-Bonsai-2 at ctx 4096: 314.8 -> 310.0 ms of this GEMM per prefill. Only the
// block->tile map changes; each tile's arithmetic is untouched. SPARKINFER_PREFILL_GEMM_I8_RASTER=0
// keeps the 2-D grid.
constexpr int PF_W4_ST = 3;
constexpr int PF_W4_GRP = 8;
constexpr int PF_W4_MF = 4;        // 64 rows per warp / 16
constexpr int PF_W4_NF = 8;        // 64 cols per warp / 8
constexpr size_t PF_W4_SMEM = (size_t)PF_W4_ST * (PF_BM + PF_BN) * PF_BK;

template <bool RESID, int GRP>
__global__ __launch_bounds__(128, 2) void pf_gemm_i8_w4_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* C, int M, int N, int K) {
    extern __shared__ __align__(16) signed char pf_w4_smem[];
    auto As = reinterpret_cast<signed char (*)[PF_BM][PF_BK]>(pf_w4_smem);
    auto Bs = reinterpret_cast<signed char (*)[PF_BN][PF_BK]>(pf_w4_smem + PF_W4_ST * PF_BM * PF_BK);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;
    const int tig  = lane & 3;
    const int sub  = lane >> 3;
    const int lrow = lane & 7;
    const int wm   = warp & 1;                        // rows [wm*64, +64)
    const int wn   = warp >> 1;                       // cols [wn*64, +64)
    int mt = blockIdx.y, nt = blockIdx.x;
    if constexpr (GRP > 0) {
        const int tiles_m = M / PF_BM, tiles_n = N / PF_BN;
        const int per = GRP * tiles_n, g = blockIdx.x / per, r = blockIdx.x - g * per;
        const int fm = g * GRP, gs = min(tiles_m - fm, GRP);
        mt = fm + r % gs;
        nt = r / gs;
    }
    const int m0   = mt * PF_BM;
    const int n0   = nt * PF_BN;
    const int nk   = K / PF_BK;

    int acc[PF_W4_MF][PF_W4_NF][4];
    #pragma unroll
    for (int i = 0; i < PF_W4_MF; i++)
        #pragma unroll
        for (int j = 0; j < PF_W4_NF; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

    // 512 16B chunks per operand tile; 128 threads stage 4 A-chunks + 4 B-chunks each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 128) {
            const int r = s >> 2, k = (s & 3) << 4;
            __pipeline_memcpy_async(&As[buf][r][pf_swz(k, r)], &A[(size_t)(m0 + r) * K + k0 + k], 16);
            __pipeline_memcpy_async(&Bs[buf][r][pf_swz(k, r)], &W[(size_t)(n0 + r) * K + k0 + k], 16);
        }
    };

    // Stage t waits for its own group, then (after the barrier that retires every read of the
    // buffer it is about to refill) issues stage t+ST-1. One commit per iteration, empty at the tail,
    // so wait_prior(ST-2) always means "tile t has landed".
    #pragma unroll
    for (int s = 0; s < PF_W4_ST - 1; s++) {
        if (s < nk) stage(s, s * PF_BK);
        __pipeline_commit();
    }
    for (int t = 0; t < nk; t++) {
        __pipeline_wait_prior(PF_W4_ST - 2);
        __syncthreads();
        const int buf = t % PF_W4_ST;
        {
            const int tn = t + PF_W4_ST - 1;
            if (tn < nk) stage(tn % PF_W4_ST, tn * PF_BK);
            __pipeline_commit();
        }
        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 32) {
            unsigned af[PF_W4_MF][4], bf[PF_W4_NF][2];
            #pragma unroll
            for (int i = 0; i < PF_W4_MF; i++) {
                const int row = wm * 64 + i * 16 + (sub & 1) * 8 + lrow;
                pf_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                          &As[buf][row][pf_swz(kk + (sub >> 1) * 16, row)]);
            }
            #pragma unroll
            for (int jp = 0; jp < PF_W4_NF; jp += 2) {
                const int col = wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                pf_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                          &Bs[buf][col][pf_swz(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < PF_W4_MF; i++)
                #pragma unroll
                for (int j = 0; j < PF_W4_NF; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
    }

    // pf_gemm_i8_kernel's FULL store, same expression order.
    #pragma unroll
    for (int i = 0; i < PF_W4_MF; i++) {
        #pragma unroll
        for (int j = 0; j < PF_W4_NF; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            const float w0 = sw[gn], w1 = sw[gn + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 64 + i * 16 + grp + h * 8;
                const float s = sx[gm];
                __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                         (float)acc[i][j][h * 2 + 1] * s * w1);
                __nv_bfloat162* cp = reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]);
                if (RESID) {
                    const __nv_bfloat162 r = *cp;
                    v = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(v.x),
                                              __bfloat162float(r.y) + __bfloat162float(v.y));
                }
                *cp = v;
            }
        }
    }
}

// FFN gate+up+SwiGLU in one GEMM. A block owns 64 FFN channels of BOTH projections: B rows 0..63
// are gate channels [c0, c0+64), rows 64..127 the same up channels. Warp (wm, wn) takes gate
// channels [wn*32, +32) and the same up channels, so acc[i][j] (gate) and acc[i][j+4] (up) are the
// same (row, channel) in the same thread and the epilogue forms h with no exchange.
//
// Bit-identical to gate GEMM + up GEMM + launch_prefill_swiglu_quant_i8: g and u are rounded to
// bf16 by the same expression pf_gemm_i8_kernel's store uses, and h by the one the SwiGLU kernel
// uses (pf_gu_silu is its sq_silu; both TUs are si_fused with the same flags). h is then quantized
// by launch_prefill_quant_h_i8, which reads back exactly the value the SwiGLU kernel held.
// What it removes is the bf16 gate and up planes: 4 B written and 4 B read per FFN activation
// become 2 + 2. SPARKINFER_PREFILL_GEMM_I8_SWIGLU=0 keeps the three-kernel form.
__device__ __forceinline__ float pf_gu_silu(float x) { return x / (1.f + __expf(-x)); }

template <int GRP>
__global__ __launch_bounds__(128, 2) void pf_gemm_i8_gu_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ Wg,
        const signed char* __restrict__ Wu, const float* __restrict__ sx,
        const float* __restrict__ swg, const float* __restrict__ swu,
        __nv_bfloat16* __restrict__ Hout, int ldh, int M, int NH, int K) {
    extern __shared__ __align__(16) signed char pf_w4_smem[];
    auto As = reinterpret_cast<signed char (*)[PF_BM][PF_BK]>(pf_w4_smem);
    auto Bs = reinterpret_cast<signed char (*)[PF_BN][PF_BK]>(pf_w4_smem + PF_W4_ST * PF_BM * PF_BK);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;
    const int tig  = lane & 3;
    const int sub  = lane >> 3;
    const int lrow = lane & 7;
    const int wm   = warp & 1;                        // rows [wm*64, +64)
    const int wn   = warp >> 1;                       // channels [wn*32, +32) of gate and of up
    constexpr int CH = PF_BN / 2;                     // 64 channels per block
    int mt = blockIdx.y, nt = blockIdx.x;
    if constexpr (GRP > 0) {
        const int tiles_m = M / PF_BM, tiles_n = NH / CH;
        const int per = GRP * tiles_n, g = blockIdx.x / per, r = blockIdx.x - g * per;
        const int fm = g * GRP, gs = min(tiles_m - fm, GRP);
        mt = fm + r % gs;
        nt = r / gs;
    }
    const int m0 = mt * PF_BM;
    const int c0 = nt * CH;
    const int nk = K / PF_BK;

    int acc[PF_W4_MF][PF_W4_NF][4];
    #pragma unroll
    for (int i = 0; i < PF_W4_MF; i++)
        #pragma unroll
        for (int j = 0; j < PF_W4_NF; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 128) {
            const int r = s >> 2, k = (s & 3) << 4;
            const signed char* b = r < CH ? &Wg[(size_t)(c0 + r) * K] : &Wu[(size_t)(c0 + r - CH) * K];
            __pipeline_memcpy_async(&As[buf][r][pf_swz(k, r)], &A[(size_t)(m0 + r) * K + k0 + k], 16);
            __pipeline_memcpy_async(&Bs[buf][r][pf_swz(k, r)], b + k0 + k, 16);
        }
    };

    #pragma unroll
    for (int s = 0; s < PF_W4_ST - 1; s++) {
        if (s < nk) stage(s, s * PF_BK);
        __pipeline_commit();
    }
    for (int t = 0; t < nk; t++) {
        __pipeline_wait_prior(PF_W4_ST - 2);
        __syncthreads();
        const int buf = t % PF_W4_ST;
        {
            const int tn = t + PF_W4_ST - 1;
            if (tn < nk) stage(tn % PF_W4_ST, tn * PF_BK);
            __pipeline_commit();
        }
        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 32) {
            unsigned af[PF_W4_MF][4], bf[PF_W4_NF][2];
            #pragma unroll
            for (int i = 0; i < PF_W4_MF; i++) {
                const int row = wm * 64 + i * 16 + (sub & 1) * 8 + lrow;
                pf_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                          &As[buf][row][pf_swz(kk + (sub >> 1) * 16, row)]);
            }
            // jp 0,2: gate channels; jp 4,6: the same up channels (Bs rows +64)
            #pragma unroll
            for (int jp = 0; jp < PF_W4_NF; jp += 2) {
                const int col = (jp >= 4 ? CH : 0) + wn * 32 + ((jp & 3) + (sub >> 1)) * 8 + lrow;
                pf_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                          &Bs[buf][col][pf_swz(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < PF_W4_MF; i++)
                #pragma unroll
                for (int j = 0; j < PF_W4_NF; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
    }

    #pragma unroll
    for (int i = 0; i < PF_W4_MF; i++) {
        #pragma unroll
        for (int j = 0; j < PF_W4_NF / 2; j++) {
            const int ch = c0 + wn * 32 + j * 8 + tig * 2;
            const float g0 = swg[ch], g1 = swg[ch + 1], u0 = swu[ch], u1 = swu[ch + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 64 + i * 16 + grp + h * 8;
                const float s = sx[gm];
                // the gate and up GEMMs' stores ...
                const __nv_bfloat162 gv = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * g0,
                                                                (float)acc[i][j][h * 2 + 1] * s * g1);
                const __nv_bfloat162 uv = __floats2bfloat162_rn((float)acc[i][j + 4][h * 2] * s * u0,
                                                                (float)acc[i][j + 4][h * 2 + 1] * s * u1);
                // ... then the SwiGLU kernel's h
                __nv_bfloat162 hv;
                hv.x = __float2bfloat16(pf_gu_silu(__bfloat162float(gv.x)) * __bfloat162float(uv.x));
                hv.y = __float2bfloat16(pf_gu_silu(__bfloat162float(gv.y)) * __bfloat162float(uv.y));
                *reinterpret_cast<__nv_bfloat162*>(&Hout[(size_t)gm * ldh + ch]) = hv;
            }
        }
    }
}

// The long-prefill GEMMs run at the board's power limit (575 W, 2.0-2.45 GHz sustained), so what
// they cost is energy per MAC, not issue slots. This tile halves the barriers and the cp.async
// groups per MAC (BK=128: a 128-byte row per stage) and stages 3 bytes per 256 MACs instead of 4
// (128x256 block, eight 64x64 warps, two stages in 96 KB, one block per SM). Sustained on the
// Bonsai shapes at GRP=8: 590 -> 650 TOPS on gate/up and down, 591 -> 634 at N=12288. A 128-byte
// row puts an 8-row ldmatrix phase on one bank line, so the swizzle takes all three row bits.
//
// GU=true is pf_gemm_i8_gu_kernel at twice the width: B rows 0..127 are gate channels
// [c0, c0+128), rows 128..255 the same up channels, and warp wn takes gate channels [wn*32, +32)
// and the same up channels. Every tile is an exact int32 sum and both epilogues are copied from
// the kernels above, so C (and H) are bit-identical. SPARKINFER_PREFILL_GEMM_I8_BK128=0 keeps them.
constexpr int PF_W8_BK = 128;
constexpr int PF_W8_BN = 256;
constexpr int PF_W8_ST = 2;
// Raster group: 32 M-tiles (a whole 4096-row chunk) walk each weight tile together, so the
// weights leave DRAM once per chunk instead of four times at GRP=8. Sustained, gate/up
// 654 -> 692 TOPS at M=4096; at the 8192-row chunks 32 is within 1% of the whole-M walk.
constexpr int PF_W8_GRP = 32;
constexpr size_t PF_W8_SMEM = (size_t)PF_W8_ST * (PF_BM + PF_W8_BN) * PF_W8_BK;

__device__ __forceinline__ int pf_swz8(int k, int row) {
    return (((k >> 4) ^ (row & 7)) << 4) | (k & 15);
}

template <bool RESID, bool GU>
__global__ __launch_bounds__(256, 1) void pf_gemm_i8_w8_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ W,
        const signed char* __restrict__ Wu, const float* __restrict__ sx,
        const float* __restrict__ sw, const float* __restrict__ swu,
        __nv_bfloat16* C, int ldc, int M, int N, int K) {
    extern __shared__ __align__(16) signed char pf_w8_smem[];
    auto As = reinterpret_cast<signed char (*)[PF_BM][PF_W8_BK]>(pf_w8_smem);
    auto Bs = reinterpret_cast<signed char (*)[PF_W8_BN][PF_W8_BK]>(pf_w8_smem + PF_W8_ST * PF_BM * PF_W8_BK);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;
    const int tig  = lane & 3;
    const int sub  = lane >> 3;
    const int lrow = lane & 7;
    const int wm   = warp & 1;                        // rows [wm*64, +64)
    const int wn   = warp >> 1;                       // cols [wn*64, +64); GU: channels [wn*32, +32)
    constexpr int CH = PF_W8_BN / 2;                  // GU: 128 channels per block
    // N is the output width, or for GU the channel count; either way BN/(GU ? 2 : 1) per tile.
    const int tiles_m = M / PF_BM, tiles_n = N / (GU ? CH : PF_W8_BN);
    const int per = PF_W8_GRP * tiles_n, g = blockIdx.x / per, r = blockIdx.x - g * per;
    const int fm = g * PF_W8_GRP, gs = min(tiles_m - fm, PF_W8_GRP);
    const int m0 = (fm + r % gs) * PF_BM;
    const int n0 = (r / gs) * (GU ? CH : PF_W8_BN);
    const int nk = K / PF_W8_BK;

    int acc[PF_W4_MF][PF_W4_NF][4];
    #pragma unroll
    for (int i = 0; i < PF_W4_MF; i++)
        #pragma unroll
        for (int j = 0; j < PF_W4_NF; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

    // A: 128 rows x 8 chunks = 1024; B: 256 x 8 = 2048. 256 threads stage 4 + 8 chunks each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < PF_BM * 8; s += 256) {
            const int r = s >> 3, k = (s & 7) << 4;
            __pipeline_memcpy_async(&As[buf][r][pf_swz8(k, r)], &A[(size_t)(m0 + r) * K + k0 + k], 16);
        }
        #pragma unroll
        for (int s = tid; s < PF_W8_BN * 8; s += 256) {
            const int r = s >> 3, k = (s & 7) << 4;
            const signed char* b;
            if constexpr (GU) b = r < CH ? &W[(size_t)(n0 + r) * K] : &Wu[(size_t)(n0 + r - CH) * K];
            else              b = &W[(size_t)(n0 + r) * K];
            __pipeline_memcpy_async(&Bs[buf][r][pf_swz8(k, r)], b + k0 + k, 16);
        }
    };

    // Two stages: wait for tile t, barrier (retires every read of the other buffer), refill it.
    stage(0, 0);
    __pipeline_commit();
    for (int t = 0; t < nk; t++) {
        __pipeline_wait_prior(0);
        __syncthreads();
        const int buf = t & 1;
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PF_W8_BK);
        __pipeline_commit();
        #pragma unroll
        for (int kk = 0; kk < PF_W8_BK; kk += 32) {
            unsigned af[PF_W4_MF][4], bf[PF_W4_NF][2];
            #pragma unroll
            for (int i = 0; i < PF_W4_MF; i++) {
                const int row = wm * 64 + i * 16 + (sub & 1) * 8 + lrow;
                pf_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                          &As[buf][row][pf_swz8(kk + (sub >> 1) * 16, row)]);
            }
            #pragma unroll
            for (int jp = 0; jp < PF_W4_NF; jp += 2) {
                // GU: jp 0,2 gate channels; jp 4,6 the same up channels (Bs rows +128)
                const int col = GU ? (jp >= 4 ? CH : 0) + wn * 32 + ((jp & 3) + (sub >> 1)) * 8 + lrow
                                   : wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                pf_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                          &Bs[buf][col][pf_swz8(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < PF_W4_MF; i++)
                #pragma unroll
                for (int j = 0; j < PF_W4_NF; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
    }

    if constexpr (GU) {
        // pf_gemm_i8_gu_kernel's epilogue, same expression order.
        #pragma unroll
        for (int i = 0; i < PF_W4_MF; i++) {
            #pragma unroll
            for (int j = 0; j < PF_W4_NF / 2; j++) {
                const int ch = n0 + wn * 32 + j * 8 + tig * 2;
                const float g0 = sw[ch], g1 = sw[ch + 1], u0 = swu[ch], u1 = swu[ch + 1];
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int gm = m0 + wm * 64 + i * 16 + grp + h * 8;
                    const float s = sx[gm];
                    const __nv_bfloat162 gv = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * g0,
                                                                    (float)acc[i][j][h * 2 + 1] * s * g1);
                    const __nv_bfloat162 uv = __floats2bfloat162_rn((float)acc[i][j + 4][h * 2] * s * u0,
                                                                    (float)acc[i][j + 4][h * 2 + 1] * s * u1);
                    __nv_bfloat162 hv;
                    hv.x = __float2bfloat16(pf_gu_silu(__bfloat162float(gv.x)) * __bfloat162float(uv.x));
                    hv.y = __float2bfloat16(pf_gu_silu(__bfloat162float(gv.y)) * __bfloat162float(uv.y));
                    *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * ldc + ch]) = hv;
                }
            }
        }
    } else {
        // pf_gemm_i8_kernel's FULL store, same expression order.
        #pragma unroll
        for (int i = 0; i < PF_W4_MF; i++) {
            #pragma unroll
            for (int j = 0; j < PF_W4_NF; j++) {
                const int gn = n0 + wn * 64 + j * 8 + tig * 2;
                const float w0 = sw[gn], w1 = sw[gn + 1];
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int gm = m0 + wm * 64 + i * 16 + grp + h * 8;
                    const float s = sx[gm];
                    __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                             (float)acc[i][j][h * 2 + 1] * s * w1);
                    __nv_bfloat162* cp = reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * ldc + gn]);
                    if (RESID) {
                        const __nv_bfloat162 r = *cp;
                        v = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(v.x),
                                                  __bfloat162float(r.y) + __bfloat162float(v.y));
                    }
                    *cp = v;
                }
            }
        }
    }
}

// Split-K epilogue: apply the per-token / per-channel scales to the int32 partial sum. The
// arithmetic is copied from the in-GEMM epilogue above -- ((float)acc * sx[m]) * sw[n] rounded
// round-to-nearest, and for RESID the same round-then-add-then-round the fused store does -- so
// the split path and the single-block path emit the same bits.
template <bool RESID>
__global__ void pf_gemm_i8_sk_epi_kernel(const int* __restrict__ P, const float* __restrict__ sx,
                                         const float* __restrict__ sw,
                                         __nv_bfloat16* __restrict__ C, int M, int N) {
    const int m = blockIdx.y;
    if (m >= M) return;
    const float s = sx[m];
    const size_t row = (size_t)m * N;
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (n + 1 < N) {                                     // pair: one int2 load, one bf16x2 store
        const int2 p = *reinterpret_cast<const int2*>(&P[row + n]);
        __nv_bfloat162 v = __floats2bfloat162_rn((float)p.x * s * sw[n], (float)p.y * s * sw[n + 1]);
        __nv_bfloat162* cp = reinterpret_cast<__nv_bfloat162*>(&C[row + n]);
        if (RESID) {
            const __nv_bfloat162 r = *cp;
            v = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(v.x),
                                      __bfloat162float(r.y) + __bfloat162float(v.y));
        }
        *cp = v;
    } else if (n < N) {                                  // odd tail
        __nv_bfloat16 v = __float2bfloat16((float)P[row + n] * s * sw[n]);
        if (RESID)
            v = __float2bfloat16(__bfloat162float(C[row + n]) + __bfloat162float(v));
        C[row + n] = v;
    }
}

// How many ways to split K. The launch is one 128x128 output tile per block, so a projection with a
// narrow n_out gets a grid far smaller than the device: Muse Glimmer's attn k/v (n_out=256) run TWO
// blocks, and measured on an RTX 5090 that launch costs the same 69 us as the 32-block q/gate one --
// per-block streaming rate is ~12.3 GB/s and the device only saturates (~1.0 TB/s of weight) past
// ~80 blocks. Splitting K until the grid reaches that knee is what converts the idle SMs into
// throughput. Above it, extra blocks buy nothing and the partial-buffer traffic is a pure cost, so
// wide projections (Muse's ffn gate/up at 156 tiles) are deliberately left alone.
constexpr int PF_SK_TILES_MAX = 96;    // tile counts at/above this already fill the device
constexpr int PF_SK_TARGET    = 170;   // blocks to aim for (one per SM)
constexpr int PF_SK_MIN_KT    = 2;     // never leave a block fewer than this many BK-tiles
constexpr int PF_SK_MAX       = 32;    // cap: partial-buffer atomics scale with the split count

static int pf_sk_splits(int M, int N, int K) {
    const int tiles = ((N + PF_BN - 1) / PF_BN) * ((M + PF_BM - 1) / PF_BM);
    if (tiles <= 0 || tiles >= PF_SK_TILES_MAX) return 1;
    int s = (PF_SK_TARGET + tiles - 1) / tiles;
    if (s > PF_SK_MAX) s = PF_SK_MAX;
    const int smax = ((K + PF_BK - 1) / PF_BK) / PF_SK_MIN_KT;
    if (s > smax) s = smax;
    return s > 1 ? s : 1;
}
} // namespace

bool launch_prefill_gemm_i8_splitk(const signed char* A, const signed char* W,
                                   const float* sx, const float* sw, void* C,
                                   int M, int N, int K, int* partials, bool resid,
                                   cudaStream_t stream) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GEMM_SPLITK");
        return !(e && e[0] == '0');
    }();
    // One M tile only: the partial buffer is M*N int32 and the caller sizes it for that.
    if (!on || !partials || M <= 0 || M > PF_BM || N <= 0 || K <= 0) return false;
    const int splits = pf_sk_splits(M, N, K);
    if (splits <= 1) return false;
    const int nk = (K + PF_BK - 1) / PF_BK;
    const int ktiles = (nk + splits - 1) / splits;
    const int nz = (nk + ktiles - 1) / ktiles;
    if (cudaMemsetAsync(partials, 0, (size_t)M * N * sizeof(int), stream) != cudaSuccess) return false;
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM, nz);
    pf_gemm_i8_kernel<false, true><<<grid, 256, 0, stream>>>(
        A, W, sx, sw, nullptr, partials, M, N, K, ktiles);
    dim3 eg(((N + 1) / 2 + 255) / 256, M);
    if (resid)
        pf_gemm_i8_sk_epi_kernel<true><<<eg, 256, 0, stream>>>(
            partials, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N);
    else
        pf_gemm_i8_sk_epi_kernel<false><<<eg, 256, 0, stream>>>(
            partials, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N);
    return true;
}

bool launch_prefill_quantize_rows_i8(const void* x_bf16, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream, signed char* qp) {
    // Block-parallel single-pass path (one block per row, row held in registers; bit-identical).
    // SPARKINFER_PREFILL_QUANT_ROWS=0 restores the warp-per-row kernel below.
    if (launch_prefill_quant_rows_fast(x_bf16, q, scale, rows, cols, stream, qp)) return true;
    pf_quantize_rows_i8<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), q, scale, rows, cols);
    // The warp-per-row fallback has no k-tiled output, so tell the caller its packed copy is stale.
    return qp == nullptr;
}

static bool pf_gemm_i8_aligned_on() {
    static int e = -1;
    if (e < 0) {
        const char* v = getenv("SPARKINFER_PREFILL_GEMM_I8_ALIGNED");
        e = (v && v[0] == '0') ? 0 : 1;
    }
    return e != 0;
}
static bool pf_gemm_i8_full_tile(int M, int N, int K) {
    return pf_gemm_i8_aligned_on() && M > 0 && N > 0 && K > 0 &&
           (M % PF_BM) == 0 && (N % PF_BN) == 0 && (K % PF_BK) == 0;
}

static bool pf_gemm_i8_env_on(const char* name) {
    const char* v = getenv(name);
    return !(v && v[0] == '0');
}
static bool pf_gemm_i8_swz_on() {
    static const bool on = pf_gemm_i8_env_on("SPARKINFER_PREFILL_GEMM_I8_SWZ");
    return on;
}

// pf_gemm_i8_w8_kernel where the shape tiles it exactly (M % 128, N % 256 -- for GU the channel
// count % 128 -- and K % 128); false launches nothing and the caller takes its own kernel.
template <bool RESID, bool GU>
static bool pf_gemm_i8_w8_run(const signed char* A, const signed char* W, const signed char* Wu,
                              const float* sx, const float* sw, const float* swu,
                              __nv_bfloat16* c, int ldc, int M, int N, int K, cudaStream_t stream) {
    static const bool on = pf_gemm_i8_env_on("SPARKINFER_PREFILL_GEMM_I8_BK128");
    const int bn = GU ? PF_W8_BN / 2 : PF_W8_BN;
    if (!on || M <= 0 || N <= 0 || K <= 0 || (M % PF_BM) || (N % bn) || (K % PF_W8_BK)) return false;
    // One block per SM: a grid short of one wave leaves SMs the 128x128 tile would have used.
    static const int sms = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n;
    }();
    if ((M / PF_BM) * (N / bn) < sms) return false;
    static const bool attr = [] {
        cudaFuncSetAttribute(pf_gemm_i8_w8_kernel<RESID, GU>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PF_W8_SMEM);
        cudaFuncSetAttribute(pf_gemm_i8_w8_kernel<RESID, GU>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)attr;
    pf_gemm_i8_w8_kernel<RESID, GU><<<(M / PF_BM) * (N / bn), 256, PF_W8_SMEM, stream>>>(
        A, W, Wu, sx, sw, swu, c, ldc, M, N, K);
    return true;
}

template <bool RESID>
static void pf_gemm_i8_run(const signed char* A, const signed char* W, const float* sx,
                           const float* sw, __nv_bfloat16* c, int M, int N, int K,
                           cudaStream_t stream) {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    const bool full = pf_gemm_i8_full_tile(M, N, K);
    if (full && pf_gemm_i8_swz_on() &&
        pf_gemm_i8_w8_run<RESID, false>(A, W, nullptr, sx, sw, nullptr, c, N, M, N, K, stream))
        return;
    static const bool w4 = pf_gemm_i8_env_on("SPARKINFER_PREFILL_GEMM_I8_W4");
    if (full && w4 && pf_gemm_i8_swz_on()) {
        static const bool raster = pf_gemm_i8_env_on("SPARKINFER_PREFILL_GEMM_I8_RASTER");
        static const bool attr = [] {
            // Ask for the full carveout so two 48 KB blocks share an SM.
            cudaFuncSetAttribute(pf_gemm_i8_w4_kernel<RESID, PF_W4_GRP>,
                                 cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            cudaFuncSetAttribute(pf_gemm_i8_w4_kernel<RESID, 0>,
                                 cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            return true;
        }();
        (void)attr;
        if (raster)
            pf_gemm_i8_w4_kernel<RESID, PF_W4_GRP><<<dim3(grid.x * grid.y), 128, PF_W4_SMEM, stream>>>(
                A, W, sx, sw, c, M, N, K);
        else
            pf_gemm_i8_w4_kernel<RESID, 0><<<grid, 128, PF_W4_SMEM, stream>>>(A, W, sx, sw, c, M, N, K);
    } else if (!pf_gemm_i8_swz_on()) {
        if (full)
            pf_gemm_i8_kernel<RESID, false, true, 0><<<grid, 256, 0, stream>>>(
                A, W, sx, sw, c, nullptr, M, N, K, 0);
        else
            pf_gemm_i8_kernel<RESID, false, false, 0><<<grid, 256, 0, stream>>>(
                A, W, sx, sw, c, nullptr, M, N, K, 0);
    } else if (full) {
        pf_gemm_i8_kernel<RESID, false, true><<<grid, 256, 0, stream>>>(
            A, W, sx, sw, c, nullptr, M, N, K, 0);
    } else {
        pf_gemm_i8_kernel<RESID, false, false><<<grid, 256, 0, stream>>>(
            A, W, sx, sw, c, nullptr, M, N, K, 0);
    }
}

void launch_prefill_gemm_i8(const signed char* A, const signed char* W,
                            const float* sx, const float* sw, void* C,
                            int M, int N, int K, cudaStream_t stream) {
    pf_gemm_i8_run<false>(A, W, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K, stream);
}

// Residual-fused variant: C[m,n] += bf16(acc*sx*sw) with pf_add's rounding. Passing the residual
// tensor AS C removes the ao scratch round-trip and the separate full-tensor add launch.
void launch_prefill_gemm_i8_resid(const signed char* A, const signed char* W,
                                  const float* sx, const float* sw, void* C,
                                  int M, int N, int K, cudaStream_t stream) {
    pf_gemm_i8_run<true>(A, W, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K, stream);
}


bool launch_prefill_gemm_i8_swiglu(const signed char* A, const signed char* Wg,
                                   const signed char* Wu, const float* sx, const float* swg,
                                   const float* swu, void* H, int ldh, int M, int NH, int K,
                                   cudaStream_t stream) {
    // SPARKINFER_PREFILL_GEMM_I8_SWZ=0 is the full rollback to main's GEMM, so it declines here too.
    static const bool on = pf_gemm_i8_env_on("SPARKINFER_PREFILL_GEMM_I8_SWIGLU");
    if (!on || !pf_gemm_i8_swz_on() || M <= 0 || NH <= 0 || K <= 0 || (M % PF_BM) || (NH % (PF_BN / 2)) || (K % PF_BK) ||
        (ldh % 2))
        return false;
    static const bool attr = [] {
        cudaFuncSetAttribute(pf_gemm_i8_gu_kernel<PF_W4_GRP>,
                             cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return true;
    }();
    (void)attr;
    if (pf_gemm_i8_w8_run<false, true>(A, Wg, Wu, sx, swg, swu, reinterpret_cast<__nv_bfloat16*>(H),
                                       ldh, M, NH, K, stream))
        return true;
    const int tiles = (M / PF_BM) * (NH / (PF_BN / 2));
    pf_gemm_i8_gu_kernel<PF_W4_GRP><<<tiles, 128, PF_W4_SMEM, stream>>>(
        A, Wg, Wu, sx, swg, swu, reinterpret_cast<__nv_bfloat16*>(H), ldh, M, NH, K);
    return true;
}

}} // namespace sparkinfer::kernels

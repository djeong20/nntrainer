#define QK_K 256
#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1
#define QK8_1 32
#define K_SCALE_SIZE 12
#define WARP_SIZE 32
#define VDR_Q4_K_Q8_1_MMQ  8
#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE/QI8_1)
#define MMQ_ITER_K 256

#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

struct tile_x_sizes {
    int qs;
    int dm;
    int sc;
};

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
                        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4*QK8_1]; // 128 values quantized to 8 bit each
};

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr tile_x_sizes txs{mmq_y*WARP_SIZE + mmq_y, mmq_y*WARP_SIZE/QI4_K, mmq_y*WARP_SIZE/8 + mmq_y/8};
    int   * x_qs = (int   *)  x_tile; // q4_K quants
    half2 * x_dm = (half2 *) (x_qs + txs.qs); // q4_K super-block scale for quantized scales/mins
    int   * x_sc = (int   *) (x_dm + txs.dm); // q4_K scales and mins, quantized with 6 bits

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;
        const int qs0 = get_int_b4(bxi->qs, threadIdx.x);

        x_qs[i*(WARP_SIZE + 1) + threadIdx.x] = qs0;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps*QI4_K) {
        int i = (i0 + threadIdx.y*QI4_K + threadIdx.x) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride;

        x_dm[i] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + threadIdx.y * 8 + threadIdx.x / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *) x + kbx0 + i*stride + (threadIdx.x % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (WARP_SIZE/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(WARP_SIZE/8) + i/8 + ksc] = scales8;
    }
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void mmq_write_back_dp4a(
        const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,
        const int stride, const int i_max, const int j_max) {
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        if (j > j_max) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            if (need_check && i > i_max) {
                continue;
            }

            dst[ids_dst[j]*stride + i] = sum[(j0/nwarps) * (mmq_y/WARP_SIZE) + i0/WARP_SIZE];
        }
    }
}


static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            /// @todo: this should be dot(uchar4 a, char4 b) from __opencl_c_integer_dot_product_input_4x8bit
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }
        // half2 type to float2 type conversion
        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    // half2 type to float2 type conversion
    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}


template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_dp4a(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    constexpr tile_x_sizes txs{mmq_y*WARP_SIZE + mmq_y, mmq_y*WARP_SIZE/QI4_K, mmq_y*WARP_SIZE/8 + mmq_y/8};
    const int   * x_qs = (const int   *) x; // q4_K quants
    const half2 * x_dm = (const half2 *) x_qs + txs.qs; // q4_K super-block scales for quantized scales/mins
    const int   * x_sc = (const int   *) x_dm + txs.dm; // q4_K scales and mins, quantized with 6 bits

    const int   * y_qs = (const int   *) y + 4; // q8_1 quants, skip first 4 bytes (q8_1 header)
    const half2 * y_ds = (const half2 *) y; // q8_1 deltas and sums

// #pragma unroll
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k0/32] + 2*(k01/16);

                sum[j0/nwarps*mmq_y/WARP_SIZE + i0/WARP_SIZE] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(WARP_SIZE + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}


/**
 * @brief This is a main kernel for Q4_K x Q8_1 matrix multiplication.
 */
template <int nwarps, bool need_check, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
        const char * __restrict__ x, const int offset_x, const int * __restrict__ y,
        const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int tile_x_max_i, const int tile_y_max_j, const int kb0_start, const int kb0_stop) {

    constexpr int qk         = QK_K;
    constexpr int mmq_x      = 64; // Either 64 or 128: get_mmq_x_device()
    constexpr int mmq_y      = 128; // Either 64 or 128: get_mmq_y_device()

    // shared memory to store Q8_1 tiles
    extern __shared__ int data_mul_mat_q[];
    int * tile_y = data_mul_mat_q + mmq_x;
    int * tile_x = tile_y + GGML_PAD(mmq_x*(WARP_SIZE + WARP_SIZE/QI8_1), nwarps*WARP_SIZE);

    constexpr int blocks_per_iter = MMQ_ITER_K / qk;

    float sum[mmq_x*mmq_y / (nwarps*WARP_SIZE)] = {0.0f};

    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
        load_tiles_q4_K(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

        {
            const int * by0 = y + ncols_y*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 0*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot_q4_K_q8_1_dp4a(tile_x, tile_y, sum, 0);

        __syncthreads();

        {
            const int * by0 = y + ncols_y*(kb0*(qk*sizeof(block_q8_1_mmq) / (4*QK8_1*sizeof(int))) + 1*sizeof(block_q8_1_mmq)/sizeof(int));
#pragma unroll
            for (int l0 = 0; l0 < mmq_x*MMQ_TILE_Y_K; l0 += nwarps*WARP_SIZE) {
                int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                tile_y[l] = by0[l];
            }
        }

        __syncthreads();

        vec_dot_q4_K_q8_1_dp4a(tile_x, tile_y, sum, WARP_SIZE);

        __syncthreads();
    }

    if (fixup) {
        mmq_write_back_dp4a(sum, ids_dst, tmp_fixup + blockIdx.x*(mmq_x*mmq_y), mmq_y, mmq_y, mmq_x);
    } else {
        mmq_write_back_dp4a(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

template <ggml_type type, int mmq_x, int nwarps, bool need_check>
static __global__ void mul_mat_q(
        const char * __restrict__ x, const int * __restrict__ y, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, float * __restrict__ dst, float * __restrict__ tmp_fixup,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int channel_ratio, const int nchannels_y, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int sample_ratio, const int nsamples_y, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {

    constexpr int qk    = QK_K;
    constexpr int mmq_x = 64; // Either 64 or 128: get_mmq_x_device()
    constexpr int mmq_y = 128; // Either 64 or 128: get_mmq_y_device()

    const int ntx = (ncols_dst + mmq_x - 1) / mmq_x; // Number of tiles x
    const int nty = (nrows_x   + mmq_y - 1) / mmq_y; // Number of tiles y

    // Initialize the ids for writing back data with just the index.
    // For regular matrix multiplications this is never changed.
    // For MoE the correct indices are loaded from ids_dst.
    extern __shared__ int ids_dst_shared[]; // Stored at beginning of shared memory.
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*WARP_SIZE) {
        const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

        if (j0 + nwarps*WARP_SIZE > mmq_x && j >= mmq_x) {
            break;
        }

        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const     int64_t blocks_per_ne00 = ncols_x / qk;
    constexpr int     blocks_per_iter = MMQ_ITER_K / qk;

    // kbc == k block continuous, current index in continuous ijk space.
    int64_t kbc      = (int64_t) blockIdx.x     *nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;
    int64_t kbc_stop = (int64_t)(blockIdx.x + 1)*nsamples_y*nchannels_y*ntx*nty*blocks_per_ne00 / gridDim.x;

    kbc      -= (kbc      % blocks_per_ne00) % blocks_per_iter;
    kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

    // kb0 == k index when doing the matrix multiplication for an output tile.
    int kb0_start = kbc % blocks_per_ne00;
    int kb0_stop  = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
        int tmp = kbc;
        const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
        tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
        const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
        tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
        const int zt = tmp / (ntx*blocks_per_ne00);
        tmp -= zt * (ntx*blocks_per_ne00);
        const int jt = tmp / blocks_per_ne00;

        // Defaults for regular matrix multiplication:
        int col_low    = 0;
        int col_high   = ncols_dst;
        int col_diff   = ncols_dst;
        int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
        int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

        if (ids_dst) {
            col_low  = expert_bounds[zt + 0];
            col_high = expert_bounds[zt + 1];
            col_diff = col_high - col_low;

            offset_y   = 0;
            offset_dst = 0;

            if (jt*mmq_x >= col_diff) {
                kbc += blocks_per_ne00;
                kbc -= kbc % blocks_per_ne00;

                kb0_start = 0;
                kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);

                continue;
            }

            __syncthreads();
#pragma unroll
            for (int j0 = 0; j0 < mmq_x; j0 += nwarps*WARP_SIZE) {
                const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

                if (j0 + nwarps*WARP_SIZE > mmq_x && j >= mmq_x) {
                    break;
                }

                ids_dst_shared[j] = ids_dst[col_low + jt*mmq_x + j];
            }
            __syncthreads();
        }

        offset_y   += (col_low + jt*mmq_x)*(sizeof(block_q8_1_mmq)/sizeof(int));
        offset_dst += it*mmq_y;

        const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
        const int tile_y_max_j = col_diff - jt*mmq_x - 1;

        const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

        constexpr bool fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        mul_mat_q_process_tile<nwarps, need_check, fixup>
            (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
             tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);

        kbc += blocks_per_ne00;
        kbc -= kbc % blocks_per_ne00;

        kb0_start = 0;
        kb0_stop  = min(blocks_per_ne00, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    int tmp = kbc;
    const int it = tmp / (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
    tmp -= it * (nsamples_y*nchannels_y*ntx*blocks_per_ne00);
    const int wt = tmp / (nchannels_y*ntx*blocks_per_ne00);
    tmp -= wt * (nchannels_y*ntx*blocks_per_ne00);
    const int zt = tmp / (ntx*blocks_per_ne00);
    tmp -= zt * (ntx*blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    // Defaults for regular matrix multiplication:
    int col_low    = 0;
    int col_high   = ncols_dst;
    int col_diff   = ncols_dst;
    int offset_y   = wt*stride_sample_y   + zt*stride_channel_y;
    int offset_dst = wt*stride_sample_dst + zt*stride_channel_dst + jt*mmq_x*stride_col_dst;

    if (ids_dst) {
        col_low  = expert_bounds[zt + 0];
        col_high = expert_bounds[zt + 1];
        col_diff = col_high - col_low;

        offset_y   = 0;
        offset_dst = 0;

        if (jt*mmq_x >= col_diff) {
            return;
        }

        // The memory layout for the fixup buffer is always contiguous, therefore reset ids:
        __syncthreads();
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps*WARP_SIZE) {
            const int j = j0 + threadIdx.y*WARP_SIZE + threadIdx.x;

            if (j0 + nwarps*WARP_SIZE > mmq_x && j >= mmq_x) {
                break;
            }

            ids_dst_shared[j] = j;
        }
        __syncthreads();
    }

    offset_y   += (col_low + jt*mmq_x)*(sizeof(block_q8_1_mmq)/sizeof(int));
    offset_dst += it*mmq_y;

    const int tile_x_max_i = nrows_x  - it*mmq_y - 1;
    const int tile_y_max_j = col_diff - jt*mmq_x - 1;

    const int offset_x = (wt/sample_ratio)*stride_sample_x + (zt/channel_ratio)*stride_channel_x + it*mmq_y*stride_row_x;

    constexpr bool fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    mul_mat_q_process_tile<nwarps, need_check, fixup>
        (x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup, stride_row_x, ncols_y, stride_col_dst,
         tile_x_max_i, tile_y_max_j, kb0_start, kb0_stop);
}

/// launch_mul_mat_q() is the main entry point for the kernel. It should be implemenated the blas_kernel.cpp
/// https://github.com/ggml-org/llama.cpp/blob/8f974bc1e980c06833504276021072e7a4088c81/ggml/src/ggml-cuda/mmq.cuh#L3009

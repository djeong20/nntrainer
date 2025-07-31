#define VDR_Q4_K_Q8_1_MMQ 8

#define QK_K 256
#define K_SCALE_SIZE 12
#define WARP_SIZE 16

#define QI4_K (QK_K / (4 * QR4_K))
#define QR4_K 2

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QK8_1 32

#if defined(SYCL_USE_XMX)
#define MMQ_X_Q4_K_AMPERE 4
#define MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define MMQ_X_Q4_K_AMPERE 64
#define MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif

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
  uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;

typedef struct {
  GGML_EXTENSION union {
    struct {
      ggml_half d; // delta
      ggml_half s; // d * sum(qs[i])
    } GGML_COMMON_AGGR_S;
    ggml_half2 ds;
  } GGML_COMMON_AGGR_U;
  int8_t qs[QK8_1]; // quants
} block_q8_1;

static __dpct_inline__ int get_int_from_uint8_aligned(const uint8_t *x8,
                                                      const int &i32) {
  return *(
    (const int *)(x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __dpct_inline__ void allocate_tiles_q4_K(int **x_ql, sycl::half2 **x_dm,
                                                int **x_qh, int **x_sc,
                                                int *tile_x_ql_q4_K,
                                                sycl::half2 *tile_x_dm_q4_K,
                                                int *tile_x_sc_q4_K) {
  (void)x_qh;

  *x_ql = tile_x_ql_q4_K; // quants
  *x_dm = tile_x_dm_q4_K; // delta
  *x_sc = tile_x_sc_q4_K; // quants
}

static __dpct_inline__ void
load_tiles_q4_K(const void *__restrict__ vx, int *__restrict__ x_ql,
                sycl::half2 *__restrict__ x_dm, int *__restrict__ x_qh,
                int *__restrict__ x_sc, const int &i_offset, const int &i_max,
                const int &k, const int &blocks_per_row) {
  (void)x_qh;

  const int kbx = k / QI4_K;  // == 0 if QK_K == 256
  const int kqsx = k % QI4_K; // == k if QK_K == 256

  const block_q4_K *bx0 = (const block_q4_K *)vx;

#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y_Q4_K_AMPERE; i0 += NWARPS_Q4_K_AMPERE) {
    int i = i0 + i_offset;

    const block_q4_K *bxi = bx0 + i * blocks_per_row + kbx;

    x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
  }

  constexpr int blocks_per_tile_x_row =
    QI4_K > WARP_SIZE ? 1 : WARP_SIZE / QI4_K; // == 1 if QK_K == 256
  const int kbxd = k % blocks_per_tile_x_row;  // == 0 if QK_K == 256

#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y_Q4_K_AMPERE; i0 += NWARPS_Q4_K_AMPERE * QI4_K) {
    int i =
      (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % MMQ_Y_Q4_K_AMPERE;

    const block_q4_K *bxi = bx0 + i * blocks_per_row + kbxd;

    x_dm[i * (WARP_SIZE / QI4_K) + i / QI4_K + kbxd] = bxi->dm;
  }

#pragma unroll
  for (int i0 = 0; i0 < MMQ_Y_Q4_K_AMPERE; i0 += NWARPS_Q4_K_AMPERE * 8) {
    int i = (i0 + i_offset * 8 + k / (WARP_SIZE / 8)) % MMQ_Y_Q4_K_AMPERE;

    const block_q4_K *bxi =
      bx0 + i * blocks_per_row + (k % (WARP_SIZE / 8)) / (QI4_K / 8);

    const int *scales = (const int *)bxi->scales;

    const int ksc = k % (WARP_SIZE / 8);

    // scale arrangement after the following two lines: sc0,...,sc3,
    // sc4,...,sc7, m0,...,m3, m4,...,m8
    int scales8 = (scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) &
                  0x0F0F0F0F; // lower 4 bits
    scales8 |=
      (scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030; // upper 2 bits

    x_sc[i * (WARP_SIZE / 8) + i / 8 + ksc] = scales8;
  }
}

// contiguous u/y values
static __dpct_inline__ float vec_dot_q4_K_q8_1_impl_mmq(
  const int *__restrict__ v, const int *__restrict__ u,
  const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
  const sycl::half2 &dm4, const sycl::half2 *__restrict__ ds8) {

  float sumf_d = 0.0f;
  float sumf_m = 0.0f;

#pragma unroll
  for (int i = 0; i < QR4_K * VDR_Q4_K_Q8_1_MMQ / QI8_1; ++i) {
    int sumi_d = 0;

#pragma unroll
    for (int j = 0; j < QI8_1; ++j) {
      sumi_d = dpct::dp4a((v[j] >> (4 * i)) & 0x0F0F0F0F, u[i * QI8_1 + j],
                          sumi_d); // SIMD dot product
    }

    const sycl::float2 ds8f =
      ds8[i].convert<float, sycl::rounding_mode::automatic>();

    sumf_d += ds8f.x() * (sc[i] * sumi_d);
    sumf_m += ds8f.y() * m[i]; // sum of q8_1 block * q4_K min val
  }

  const sycl::float2 dm4f =
    dm4.convert<float, sycl::rounding_mode::automatic>();

  return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

static __dpct_inline__ float vec_dot_q4_K_q8_1_mul_mat(
  const int *__restrict__ x_ql, const sycl::half2 *__restrict__ x_dm,
  const int *__restrict__ x_qh, const int *__restrict__ x_sc,
  const int *__restrict__ y_qs, const sycl::half2 *__restrict__ y_ds,
  const int &i, const int &j, const int &k) {
  (void)x_qh;

  const uint8_t *sc =
    ((const uint8_t *)&x_sc[i * (WARP_SIZE / 8) + i / 8 + k / 16]) +
    2 * ((k % 16) / 8);

  const int index_y = j * WARP_SIZE + (QR4_K * k) % WARP_SIZE;
  return vec_dot_q4_K_q8_1_impl_mmq(
    &x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc + 8,
    x_dm[i * (WARP_SIZE / QI4_K) + i / QI4_K], &y_ds[index_y / QI8_1]);
}

static __dpct_inline__ void
mul_mat_q(const void *__restrict__ vx, const void *__restrict__ vy,
          float *__restrict__ dst, const int ncols_x, const int nrows_x,
          const int ncols_y, const int nrows_y, const int nrows_dst,
          int *tile_x_ql, sycl::half2 *tile_x_dm, int *tile_x_qh,
          int *tile_x_sc, const sycl::nd_item<3> &item_ct1, int *tile_y_qs,
          sycl::half2 *tile_y_ds) {

  const block_q4_K *x = (const block_q4_K *)vx;
  const block_q8_1 *y = (const block_q8_1 *)vy;

  const int blocks_per_row_x = ncols_x / QK_K;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE / QI4_K;

  const int &ncols_dst = ncols_y;

  const int row_dst_0 = item_ct1.get_group(2) * MMQ_Y_Q4_K_AMPERE;
  const int &row_x_0 = row_dst_0;

  const int col_dst_0 = item_ct1.get_group(1) * MMQ_X_Q4_K_AMPERE;
  const int &col_y_0 = col_dst_0;

  float sum[MMQ_Y_Q4_K_AMPERE / WARP_SIZE]
           [MMQ_X_Q4_K_AMPERE / NWARPS_Q4_K_AMPERE] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

    load_tiles_q4_K(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
                    tile_x_qh, tile_x_sc, item_ct1.get_local_id(1),
                    nrows_x - row_x_0 - 1, item_ct1.get_local_id(2),
                    blocks_per_row_x);

#pragma unroll
    for (int ir = 0; ir < QR4_K; ++ir) {
      const int kqs = ir * WARP_SIZE + item_ct1.get_local_id(2);
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < MMQ_X_Q4_K_AMPERE; i += NWARPS_Q4_K_AMPERE) {
        const int col_y_eff =
          dpct::min((unsigned int)(col_y_0 + item_ct1.get_local_id(1) + i),
                    ncols_y - 1); // to prevent out-of-bounds memory accesses

        const block_q8_1 *by0 =
          &y[col_y_eff * blocks_per_col_y + ib0 * (QK_K / QK8_1) + kbxd];

        const int index_y =
          (item_ct1.get_local_id(1) + i) * WARP_SIZE + kqs % WARP_SIZE;
        tile_y_qs[index_y] =
          get_int_from_int8_aligned(by0->qs, item_ct1.get_local_id(2) % QI8_1);
      }

#pragma unroll
      for (int ids0 = 0; ids0 < MMQ_X_Q4_K_AMPERE;
           ids0 += NWARPS_Q4_K_AMPERE * QI8_1) {
        const int ids = (ids0 + item_ct1.get_local_id(1) * QI8_1 +
                         item_ct1.get_local_id(2) / (WARP_SIZE / QI8_1)) %
                        MMQ_X_Q4_K_AMPERE;
        const int kby = item_ct1.get_local_id(2) % (WARP_SIZE / QI8_1);
        const int col_y_eff = sycl::min(col_y_0 + ids, ncols_y - 1);

        // if the sum is not needed it's faster to transform the scale to f32
        // ahead of time
        const sycl::half2 *dsi_src =
          &y[col_y_eff * blocks_per_col_y + ib0 * (QK_K / QK8_1) +
             ir * (WARP_SIZE / QI8_1) + kby]
             .ds;
        sycl::half2 *dsi_dst = &tile_y_ds[ids * (WARP_SIZE / QI8_1) + kby];
        *dsi_dst = *dsi_src;
      }

      item_ct1.barrier();

      for (int k = ir * WARP_SIZE / QR4_K; k < (ir + 1) * WARP_SIZE / QR4_K;
           k += VDR_Q4_K_Q8_1_MMQ) {
#pragma unroll
        for (int j = 0; j < MMQ_X_Q4_K_AMPERE; j += NWARPS_Q4_K_AMPERE) {
#pragma unroll
          for (int i = 0; i < MMQ_Y_Q4_K_AMPERE; i += WARP_SIZE) {
            sum[i / WARP_SIZE][j / NWARPS_Q4_K_AMPERE] +=
              vec_dot_q4_K_q8_1_mul_mat(tile_x_ql, tile_x_dm, tile_x_qh,
                                        tile_x_sc, tile_y_qs, tile_y_ds,
                                        item_ct1.get_local_id(2) + i,
                                        item_ct1.get_local_id(1) + j, k);
          }
        }
      }

      item_ct1.barrier();
    }
  }

#pragma unroll
  for (int j = 0; j < MMQ_X_Q4_K_AMPERE; j += NWARPS_Q4_K_AMPERE) {
    const int col_dst = col_dst_0 + j + item_ct1.get_local_id(1);

    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < MMQ_Y_Q4_K_AMPERE; i += WARP_SIZE) {
      const int row_dst = row_dst_0 + item_ct1.get_local_id(2) + i;

      if (row_dst >= nrows_dst) {
        continue;
      }

      dst[col_dst * nrows_dst + row_dst] =
        sum[i / WARP_SIZE][j / NWARPS_Q4_K_AMPERE];
    }
  }
}

static void mul_mat_q4_K(const void *__restrict__ vx,
                         const void *__restrict__ vy, float *__restrict__ dst,
                         const int ncols_x, const int nrows_x,
                         const int ncols_y, const int nrows_y,
                         const int nrows_dst, const sycl::nd_item<3> &item_ct1,
                         int *tile_x_ql_q4_K, sycl::half2 *tile_x_dm_q4_K,
                         int *tile_x_sc_q4_K, int *tile_y_qs,
                         sycl::half2 *tile_y_ds) {
  int *tile_x_ql = nullptr;
  sycl::half2 *tile_x_dm = nullptr;
  int *tile_x_qh = nullptr;
  int *tile_x_sc = nullptr;

  // sycl_todo: change according to hardware
  const int mmq_x = MMQ_X_Q4_K_AMPERE;
  const int mmq_y = MMQ_Y_Q4_K_AMPERE;
  const int nwarps = NWARPS_Q4_K_AMPERE;
  allocate_tiles_q4_K(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc,
                      tile_x_ql_q4_K, tile_x_dm_q4_K, tile_x_sc_q4_K);
  mul_mat_q(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst,
            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, item_ct1, tile_y_qs,
            tile_y_ds);
}

static void ggml_mul_mat_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                        float *dst, const int ncols_x,
                                        const int nrows_x, const int ncols_y,
                                        const int nrows_y, const int nrows_dst,
                                        dpct::queue_ptr stream) {
  int mmq_x, mmq_y, nwarps;

  mmq_x = MMQ_X_Q4_K_AMPERE;
  mmq_y = MMQ_Y_Q4_K_AMPERE;
  nwarps = NWARPS_Q4_K_AMPERE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const sycl::range<3> block_nums(1, block_num_y, block_num_x);
  const sycl::range<3> block_dims(1, nwarps, WARP_SIZE);

  /*
  DPCT1049:34: The work-group size passed to the SYCL kernel may exceed
  the limit. To get the device limit, query
  info::device::max_work_group_size. Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    sycl_launch(stream, [&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> tile_x_ql_q4_K_acc_ct1(
        sycl::range<1>(mmq_y * (WARP_SIZE) + mmq_y), cgh);
      sycl::local_accessor<sycl::half2, 1> tile_x_dm_q4_K_acc_ct1(
        sycl::range<1>(mmq_y * (WARP_SIZE / QI4_K) + mmq_y / QI4_K), cgh);
      sycl::local_accessor<int, 1> tile_x_sc_q4_K_acc_ct1(
        sycl::range<1>(mmq_y * (WARP_SIZE / 8) + mmq_y / 8), cgh);
      sycl::local_accessor<int, 1> tile_y_qs_acc_ct1(
        sycl::range<1>(mmq_x * WARP_SIZE), cgh);
      sycl::local_accessor<sycl::half2, 1> tile_y_ds_acc_ct1(
        sycl::range<1>(mmq_x * WARP_SIZE / QI8_1), cgh);

      sycl_parallel_for(
        cgh, sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
          mul_mat_q4_K(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y,
                       nrows_dst, item_ct1, get_pointer(tile_x_ql_q4_K_acc_ct1),
                       get_pointer(tile_x_dm_q4_K_acc_ct1),
                       get_pointer(tile_x_sc_q4_K_acc_ct1),
                       get_pointer(tile_y_qs_acc_ct1),
                       get_pointer(tile_y_ds_acc_ct1));
        });
    });
  }
}
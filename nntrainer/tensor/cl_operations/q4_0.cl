// src0_q, src0_d, src1 are transposed as a preprocessing step
// 4-bit weights are transposed in groups of 4 (unsigned short int)
// consider weights originally "next to each other", now "on top of each other"
// each fiber computes a 8x4 tile of output elements
// using unshuffled weights

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_128
#endif
kernel void kernel_mul_mat_Ab_Bi_8x8(global const ushort *src0_q, // quantized A
                                     global const half *src0_d,   // A scales
                                     global half4 *src1, // B (1d image)
                                     global float *dst,  // C
                                     int m,              // M
                                     int n,              // N with padding
                                     int k,              // K
                                     int n_no_padding    // N without padding
) {

  int m_4 = m >> 2;
  int n_4 = n >> 2;

  int gy = get_global_id(0);
  int gx = get_global_id(1);
  int gx_2 = gx << 3;

  half8 c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
  half8 B;                   // registers for activations
  half8 dequantized_weights; // registers for dequantized weights
  __global const ushort *weight_ptr = src0_q + gx_2; // pointer for weights
  __global const half *scale_ptr = src0_d + gx_2;    // pointer for scales

  for (int i = 0; i < k; i += 4) { // loop through K dimension

    B.s0123 = src1[gy * 2 + (i) * (n_4)];
    B.s4567 = src1[gy * 2 + (i) * (n_4) + 1];

    // keep (i/4) and (i/32) in parenthesis, rounds down
    // load 4 consecutive groups of 4 weights
    ushort8 bits4 = vload8(
      0, weight_ptr + (i / 4) * (m)); // (i/4) because weights grouped in 4s

    // load 4 consecutive scales
    half8 scale = vload8(
      0, scale_ptr + (i / 32) * (m)); // (i/32) because 1 scale per 32 elements

    // j=0
    dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
    dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
    dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
    dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
    dequantized_weights.s4 = ((bits4.s4 & (0x000F)) - 8) * scale.s4;
    dequantized_weights.s5 = ((bits4.s5 & (0x000F)) - 8) * scale.s5;
    dequantized_weights.s6 = ((bits4.s6 & (0x000F)) - 8) * scale.s6;
    dequantized_weights.s7 = ((bits4.s7 & (0x000F)) - 8) * scale.s7;

    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;
    c4 += B * dequantized_weights.s4;
    c5 += B * dequantized_weights.s5;
    c6 += B * dequantized_weights.s6;
    c7 += B * dequantized_weights.s7;

    // j=1
    B.s0123 = src1[gy * 2 + (i + 1) * (n_4)];
    B.s4567 = src1[gy * 2 + (i + 1) * (n_4) + 1];
    dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
    dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
    dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
    dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
    dequantized_weights.s4 = (((bits4.s4 & (0x00F0)) >> 4) - 8) * scale.s4;
    dequantized_weights.s5 = (((bits4.s5 & (0x00F0)) >> 4) - 8) * scale.s5;
    dequantized_weights.s6 = (((bits4.s6 & (0x00F0)) >> 4) - 8) * scale.s6;
    dequantized_weights.s7 = (((bits4.s7 & (0x00F0)) >> 4) - 8) * scale.s7;

    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;
    c4 += B * dequantized_weights.s4;
    c5 += B * dequantized_weights.s5;
    c6 += B * dequantized_weights.s6;
    c7 += B * dequantized_weights.s7;

    // j=2
    B.s0123 = src1[gy * 2 + (i + 2) * (n_4)];
    B.s4567 = src1[gy * 2 + (i + 2) * (n_4) + 1];
    dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
    dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
    dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
    dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
    dequantized_weights.s4 = (((bits4.s4 & (0x0F00)) >> 8) - 8) * scale.s4;
    dequantized_weights.s5 = (((bits4.s5 & (0x0F00)) >> 8) - 8) * scale.s5;
    dequantized_weights.s6 = (((bits4.s6 & (0x0F00)) >> 8) - 8) * scale.s6;
    dequantized_weights.s7 = (((bits4.s7 & (0x0F00)) >> 8) - 8) * scale.s7;

    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;
    c4 += B * dequantized_weights.s4;
    c5 += B * dequantized_weights.s5;
    c6 += B * dequantized_weights.s6;
    c7 += B * dequantized_weights.s7;

    // j=3
    B.s0123 = src1[gy * 2 + (i + 3) * (n_4)];
    B.s4567 = src1[gy * 2 + (i + 3) * (n_4) + 1];
    dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
    dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
    dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
    dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
    dequantized_weights.s4 = (((bits4.s4 & (0x0F00)) >> 12) - 8) * scale.s4;
    dequantized_weights.s5 = (((bits4.s5 & (0x0F00)) >> 12) - 8) * scale.s5;
    dequantized_weights.s6 = (((bits4.s6 & (0x0F00)) >> 12) - 8) * scale.s6;
    dequantized_weights.s7 = (((bits4.s7 & (0x0F00)) >> 12) - 8) * scale.s7;

    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;
    c4 += B * dequantized_weights.s4;
    c5 += B * dequantized_weights.s5;
    c6 += B * dequantized_weights.s6;
    c7 += B * dequantized_weights.s7;
  }

  int idx = (gy << 3) * m + (gx << 3); // vectorized store 16 elements

  // conditional check if store is to a valid location. Required when N is not a
  // multiple of 8 if statements allow registers to be reused for each store
  // provides a performance boost due to reduced register footprint, which
  // increases number of concurrent waves
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s0, c1.s0, c2.s0, c3.s0, c4.s0, c5.s0, c6.s0, c7.s0), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s1, c1.s1, c2.s1, c3.s1, c4.s1, c5.s1, c6.s1, c7.s1), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s2, c1.s2, c2.s2, c3.s2, c4.s2, c5.s2, c6.s2, c7.s2), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s3, c1.s3, c2.s3, c3.s3, c4.s3, c5.s3, c6.s3, c7.s3), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s4, c1.s4, c2.s4, c3.s4, c4.s4, c5.s4, c6.s4, c7.s4), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s5, c1.s5, c2.s5, c3.s5, c4.s5, c5.s5, c6.s5, c7.s5), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s6, c1.s6, c2.s6, c3.s6, c4.s6, c5.s6, c6.s6, c7.s6), 0,
            dst + idx);
    idx += m;
  }
  if (idx + 7 < m * n_no_padding) {
    vstore8((float8)(c0.s7, c1.s7, c2.s7, c3.s7, c4.s7, c5.s7, c6.s7, c7.s7), 0,
            dst + idx);
  }
}

// src0_q, src0_d, src1 are transposed as a preprocessing step
// 4-bit weights are transposed in groups of 4 (unsigned short int)
// consider weights originally "next to each other", now "on top of each other"
// each fiber computes a 8x4 tile of output elements
// using unshuffled weights

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_128
#endif
kernel void
kernel_mul_mat_Ab_Bi_16x4(global const ushort *src0_q, // quantized A
                          global const half *src0_d,   // A scales
                          global half4 *src1,          // B (1d image)
                          global float *dst,           // C
                          int m,                       // M
                          int n,                       // N with padding
                          int k,                       // K
                          int n_no_padding             // N without padding
) {

  int m_4 = m >> 2;
  int n_4 = n >> 2;

  int gy = get_global_id(0);
  int gx = get_global_id(1);
  int gx_2 = gx << 2;

  half16 c0 = 0, c1 = 0, c2 = 0, c3 = 0; // 8x4 output elements
  half16 B;                              // registers for activations
  half4 dequantized_weights;             // registers for dequantized weights
  __global const ushort *weight_ptr = src0_q + gx_2; // pointer for weights
  __global const half *scale_ptr = src0_d + gx_2;    // pointer for scales

  for (int i = 0; i < k; i += 4) { // loop through K dimension
    B.s0123 = src1[gy * 4 + (i) * (n_4)];
    B.s4567 = src1[gy * 4 + (i) * (n_4) + 1];
    B.s89ab = src1[gy * 4 + (i) * (n_4) + 2];
    B.scdef = src1[gy * 4 + (i) * (n_4) + 3];

    // keep (i/4) and (i/32) in parenthesis, rounds down
    // load 4 consecutive groups of 4 weights
    ushort4 bits4 = vload4(
      0, weight_ptr + (i / 4) * (m)); // (i/4) because weights grouped in 4s

    // load 4 consecutive scales
    half4 scale = vload4(
      0, scale_ptr + (i / 32) * (m)); // (i/32) because 1 scale per 32 elements

    // j=0
    dequantized_weights.s0 = ((bits4.s0 & (0x000F)) - 8) * scale.s0;
    dequantized_weights.s1 = ((bits4.s1 & (0x000F)) - 8) * scale.s1;
    dequantized_weights.s2 = ((bits4.s2 & (0x000F)) - 8) * scale.s2;
    dequantized_weights.s3 = ((bits4.s3 & (0x000F)) - 8) * scale.s3;
    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;

    // j=1
    B.s0123 = src1[gy * 4 + (i + 1) * (n_4)];
    B.s4567 = src1[gy * 4 + (i + 1) * (n_4) + 1];
    B.s89ab = src1[gy * 4 + (i + 1) * (n_4) + 2];
    B.scdef = src1[gy * 4 + (i + 1) * (n_4) + 3];

    dequantized_weights.s0 = (((bits4.s0 & (0x00F0)) >> 4) - 8) * scale.s0;
    dequantized_weights.s1 = (((bits4.s1 & (0x00F0)) >> 4) - 8) * scale.s1;
    dequantized_weights.s2 = (((bits4.s2 & (0x00F0)) >> 4) - 8) * scale.s2;
    dequantized_weights.s3 = (((bits4.s3 & (0x00F0)) >> 4) - 8) * scale.s3;
    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;

    // j=2
    B.s0123 = src1[gy * 4 + (i + 2) * (n_4)];
    B.s4567 = src1[gy * 4 + (i + 2) * (n_4) + 1];
    B.s89ab = src1[gy * 4 + (i + 2) * (n_4) + 2];
    B.scdef = src1[gy * 4 + (i + 2) * (n_4) + 3];

    dequantized_weights.s0 = (((bits4.s0 & (0x0F00)) >> 8) - 8) * scale.s0;
    dequantized_weights.s1 = (((bits4.s1 & (0x0F00)) >> 8) - 8) * scale.s1;
    dequantized_weights.s2 = (((bits4.s2 & (0x0F00)) >> 8) - 8) * scale.s2;
    dequantized_weights.s3 = (((bits4.s3 & (0x0F00)) >> 8) - 8) * scale.s3;
    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;

    // j=3
    B.s0123 = src1[gy * 4 + (i + 3) * (n_4)];
    B.s4567 = src1[gy * 4 + (i + 3) * (n_4) + 1];
    B.s89ab = src1[gy * 4 + (i + 3) * (n_4) + 2];
    B.scdef = src1[gy * 4 + (i + 3) * (n_4) + 3];

    dequantized_weights.s0 = (((bits4.s0 & (0xF000)) >> 12) - 8) * scale.s0;
    dequantized_weights.s1 = (((bits4.s1 & (0xF000)) >> 12) - 8) * scale.s1;
    dequantized_weights.s2 = (((bits4.s2 & (0xF000)) >> 12) - 8) * scale.s2;
    dequantized_weights.s3 = (((bits4.s3 & (0xF000)) >> 12) - 8) * scale.s3;
    c0 += B * dequantized_weights.s0;
    c1 += B * dequantized_weights.s1;
    c2 += B * dequantized_weights.s2;
    c3 += B * dequantized_weights.s3;
  }

  // advance by 16 rows
  int idx = (gy << 4) * m + (gx << 2); // vectorized store 16 elements

  // conditional check if store is to a valid location. Required when N is not a
  // multiple of 8 if statements allow registers to be reused for each store
  // provides a performance boost due to reduced register footprint, which
  // increases number of concurrent waves
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s0, c1.s0, c2.s0, c3.s0), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s1, c1.s1, c2.s1, c3.s1), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s2, c1.s2, c2.s2, c3.s2), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s3, c1.s3, c2.s3, c3.s3), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s4, c1.s4, c2.s4, c3.s4), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s5, c1.s5, c2.s5, c3.s5), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s6, c1.s6, c2.s6, c3.s6), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s7, c1.s7, c2.s7, c3.s7), 0, dst + idx);
  }

  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s8, c1.s8, c2.s8, c3.s8), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.s9, c1.s9, c2.s9, c3.s9), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.sa, c1.sa, c2.sa, c3.sa), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.sb, c1.sb, c2.sb, c3.sb), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.sc, c1.sc, c2.sc, c3.sc), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.sd, c1.sd, c2.sd, c3.sd), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.se, c1.se, c2.se, c3.se), 0, dst + idx);
    idx += m;
  }
  if (idx + 3 < m * n_no_padding) {
    vstore4((float4)(c0.sf, c1.sf, c2.sf, c3.sf), 0, dst + idx);
  }
}
// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}


// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m)
// different from ball-query, we look for 1-nearest neighbors.
// 0 is the dummy index.
__global__ void query_ball_nearest_point_kernel(int b, int n, int m, float radius,
                                                const float *__restrict__ new_xyz,
                                                const float *__restrict__ xyz,
                                                int *__restrict__ idx,
                                                float *__restrict__ dis) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * batch_index;
  dis += m * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;
  float radius2 = radius * radius;

  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    float min_dis = radius2;
    int candidate = -1;

    for (int k = 0; k < n; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (d2 < min_dis) {
          candidate = k;
          min_dis = d2;
        }
      }
    }
    idx[j] = candidate;
    dis[j] = min_dis;
  }
}

void query_ball_nearest_point_kernel_wrapper(int b, int n, int m, float radius,
                                             const float *new_xyz,
                                             const float *xyz, int *idx, float *dis) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_nearest_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, new_xyz, xyz, idx, dis);

  CUDA_CHECK_ERRORS();
}
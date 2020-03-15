// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: ray_start (b, m, 3)
//        ray_dir (b, m, 3)
//        points (b, n, 3)
// output: idx (b, m, n_max)
//         min_d (b, m, n_max)
//         max_d (b, m, n_max)
__global__ void ball_intersect_point_kernel(
            int b, int n, int m, float radius,
            int n_max,
            const float *__restrict__ ray_start,
            const float *__restrict__ ray_dir,
            const float *__restrict__ points,
            int *__restrict__ idx,
            float *__restrict__ min_depth,
            float *__restrict__ max_depth) {

  int batch_index = blockIdx.x;
  points += batch_index * n * 3;
  ray_start += batch_index * m * 3;
  ray_dir += batch_index * m * 3;
  idx += batch_index * m * n_max;
  min_depth += batch_index * m * n_max;
  max_depth += batch_index * m * n_max;
    
  int index = threadIdx.x;
  int stride = blockDim.x;
  float radius2 = radius * radius;

  for (int j = index; j < m; j += stride) {
    
    float x0 = ray_start[j * 3 + 0];
    float y0 = ray_start[j * 3 + 1];
    float z0 = ray_start[j * 3 + 2];
    float xw = ray_dir[j * 3 + 0];
    float yw = ray_dir[j * 3 + 1];
    float zw = ray_dir[j * 3 + 2];
  
    for (int k = 0, cnt = 0; k < n && cnt < n_max; ++k) {
      float x = points[k * 3 + 0];
      float y = points[k * 3 + 1];
      float z = points[k * 3 + 2];

      x -= x0;  y -= y0;  z -= z0; // relative position
  
      float d2 = x * x + y * y + z * z;
      float d2_proj = x * xw + y * yw + z * zw;
      float r2 = d2 - d2_proj;

      if (r2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < n_max; ++l) {
            idx[j * n_max + l] = -1;
          }
        }
        idx[j * n_max + cnt] = k;
        
        float depth = sqrt(d2_proj);
        float depth_delta = sqrt(radius2 - r2);
        
        min_depth[j * n_max + cnt] = depth - depth_delta;
        max_depth[j * n_max + cnt] = depth + depth_delta;
        ++cnt;
      }
    }
  }
}

void ball_intersect_point_kernel_wrapper(
  int b, int n, int m, float radius, int n_max,
  const float *ray_start, const float *ray_dir, const float *points,
  int *idx, float *min_depth, float *max_depth) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  ball_intersect_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, n_max, ray_start, ray_dir, points, idx, min_depth, max_depth);
  
  CUDA_CHECK_ERRORS();
}
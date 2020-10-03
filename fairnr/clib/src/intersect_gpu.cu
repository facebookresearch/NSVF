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
    
    for (int l = 0; l < n_max; ++l) {
      idx[j * n_max + l] = -1;
    }

    for (int k = 0, cnt = 0; k < n && cnt < n_max; ++k) {
      float x = points[k * 3 + 0] - x0;
      float y = points[k * 3 + 1] - y0;
      float z = points[k * 3 + 2] - z0;
      float d2 = x * x + y * y + z * z;
      float d2_proj = pow(x * xw + y * yw + z * zw, 2);
      float r2 = d2 - d2_proj;
      
      if (r2 < radius2) {
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


__global__ void aabb_intersect_point_kernel(
            int b, int n, int m, float voxelsize,
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
  float half_voxel = voxelsize * 0.5; 

  for (int j = index; j < m; j += stride) {
    for (int l = 0; l < n_max; ++l) {
      idx[j * n_max + l] = -1;
    }

    for (int k = 0, cnt = 0; k < n && cnt < n_max; ++k) {
      float f_low = 0;
      float f_high = 100000.;
      bool missed = false;
      
      for (int d = 0; d < 3; ++d) {
        
        float f_dim_low, f_dim_high, temp;
        float inv_ray_dir = 1.0 / ray_dir[j * 3 + d];
        float start = ray_start[j * 3 + d];
        float aabb = points[k * 3 + d];
      
        f_dim_low  = (aabb - half_voxel - start) * inv_ray_dir;
        f_dim_high = (aabb + half_voxel - start) * inv_ray_dir;
      
        // Make sure low is less than high
        if (f_dim_high < f_dim_low) {
          temp = f_dim_low;
          f_dim_low = f_dim_high;
          f_dim_high = temp;
        }

        // If this dimension's high is less than the low we got then we definitely missed.
        if (f_dim_high < f_low) {
          missed = true; 
          break;
        }
      
        // Likewise if the low is less than the high.
        if (f_dim_low > f_high) {
          missed = true; 
          break;
        }
          
        // Add the clip from this dimension to the previous results 
        f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
        f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
        
        if (f_low > f_high) {
          missed = true; 
          break;
        }
      }

      if (!missed){
        idx[j * n_max + cnt] = k;
        min_depth[j * n_max + cnt] = f_low;
        max_depth[j * n_max + cnt] = f_high;
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


void aabb_intersect_point_kernel_wrapper(
  int b, int n, int m, float voxelsize, int n_max,
  const float *ray_start, const float *ray_dir, const float *points,
  int *idx, float *min_depth, float *max_depth) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  aabb_intersect_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, voxelsize, n_max, ray_start, ray_dir, points, idx, min_depth, max_depth);
  
  CUDA_CHECK_ERRORS();
}


__global__ void uniform_ray_sampling_kernel(
            int b, int num_rays, 
            int max_hits,
            int max_steps,
            float step_size,
            const int *__restrict__ pts_idx,
            const float *__restrict__ min_depth,
            const float *__restrict__ max_depth,
            const float *__restrict__ uniform_noise,
            int *__restrict__ sampled_idx,
            float *__restrict__ sampled_depth,
            float *__restrict__ sampled_dists) {
  
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;

  pts_idx += batch_index * num_rays * max_hits;
  min_depth += batch_index * num_rays * max_hits;
  max_depth += batch_index * num_rays * max_hits;

  uniform_noise += batch_index * num_rays * max_steps;
  sampled_idx += batch_index * num_rays * max_steps;
  sampled_depth += batch_index * num_rays * max_steps;
  sampled_dists += batch_index * num_rays * max_steps;

  // loop over all rays
  for (int j = index; j < num_rays; j += stride) {
    int H = j * max_hits, K = j * max_steps;
    int s = 0, ucur = 0, umin = 0, umax = 0;
    float last_min_depth, last_max_depth, curr_depth;
    
    // sort all depths
    while (true) {
      if (pts_idx[H + umax] == -1 || umax == max_hits || ucur == max_steps) {
        break;  // reach the maximum
      }
      if (umin < max_hits) {
        last_min_depth = min_depth[H + umin];
      }
      if (umax < max_hits) {
        last_max_depth = max_depth[H + umax];
      }
      if (ucur < max_steps) {
        curr_depth = min_depth[H] + (float(ucur) + uniform_noise[K + ucur]) * step_size;
      }
      
      if (last_max_depth <= curr_depth && last_max_depth <= last_min_depth) {
        sampled_depth[K + s] = last_max_depth;
        sampled_idx[K + s] = pts_idx[H + umax];
        umax++; s++; continue;
      }
      if (curr_depth <= last_min_depth && curr_depth <= last_max_depth) {
        sampled_depth[K + s] = curr_depth;
        sampled_idx[K + s] = pts_idx[H + umin - 1];
        ucur++; s++; continue;
      }
      if (last_min_depth <= curr_depth && last_min_depth <= last_max_depth) {
        sampled_depth[K + s] = last_min_depth;
        sampled_idx[K + s] = pts_idx[H + umin];
        umin++; s++; continue;
      }
    }

    float l_depth, r_depth;
    int step = 0;
    for (ucur = 0, umin = 0, umax = 0; ucur < max_steps - 1; ucur++) {
      l_depth = sampled_depth[K + ucur];
      r_depth = sampled_depth[K + ucur + 1];  
      sampled_depth[K + ucur] = (l_depth + r_depth) * .5;
      sampled_dists[K + ucur] = (r_depth - l_depth);
      if (sampled_depth[K + ucur] >= min_depth[H + umin] && umin < max_hits) umin++;
      if (sampled_depth[K + ucur] >= max_depth[H + umax] && umax < max_hits) umax++;
      if (umax == max_hits || pts_idx[H + umax] == -1) break;
      if (umin - 1 == umax && sampled_dists[K + ucur] > 0) {
        sampled_depth[K + step] = sampled_depth[K + ucur];
        sampled_dists[K + step] = sampled_dists[K + ucur];
        sampled_idx[K + step] = sampled_idx[K + ucur];
        step++;
      }
    }
    for (int s = step; s < max_steps; s++) {
      sampled_idx[K + s] = -1;
    }
  }


}


void uniform_ray_sampling_kernel_wrapper(
  int b, int num_rays, int max_hits, int max_steps, float step_size,
  const int *pts_idx, const float *min_depth, const float *max_depth, const float *uniform_noise,
  int *sampled_idx, float *sampled_depth, float *sampled_dists) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  uniform_ray_sampling_kernel<<<b, opt_n_threads(num_rays), 0, stream>>>(
      b, num_rays, max_hits, max_steps, step_size, pts_idx, 
      min_depth, max_depth, uniform_noise, sampled_idx, sampled_depth, sampled_dists);
  
  CUDA_CHECK_ERRORS();
}


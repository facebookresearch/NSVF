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

  for (int j = index; j < num_rays; j += stride) {
    float depth = 0.0, delta = 0.0, left_dist = 0.0, right_dist = 0.0;
    bool done = false;
    int previous_cnt = -1;
    
    for (int l = 0; l < max_steps; ++l) {
      sampled_idx[j * max_steps + l] = -1;
    }

    for (int k = 0, cnt = 0; k < max_steps && cnt < max_hits; ++k) {

      if (pts_idx[j * max_hits + cnt] == -1) {
        done = true;
        break;
      }

      delta = step_size * uniform_noise[j * max_steps + k];   // stratified samples 
      depth = depth + delta;

      while (depth > (max_depth[j * max_hits + cnt] - min_depth[j * max_hits + cnt])) {
        depth -= (max_depth[j * max_hits + cnt] - min_depth[j * max_hits + cnt]);
        cnt += 1;
        
        if ((cnt == max_hits) || (pts_idx[j * max_hits + cnt] == -1)){
          done = true;
          break;
        }
      }

      sampled_idx[j * max_steps + k] = pts_idx[j * max_hits + cnt];
      sampled_depth[j * max_steps + k] = depth + min_depth[j * max_hits + cnt];
      
      // right distance for last step
      if (previous_cnt != -1) {
        if (cnt != previous_cnt) {
          right_dist = max_depth[j * max_hits + previous_cnt] - sampled_depth[j * max_steps + k - 1];
        } else {
          right_dist = (sampled_depth[j * max_steps + k] - sampled_depth[j * max_steps + k - 1]) / 2.0;
        }
        sampled_dists[j * max_steps + k - 1] = left_dist + right_dist;
      }
      if (done) break;

      // left distance for current step
      if (cnt != -1) {
        if (cnt != previous_cnt) {  // cross voxel boundary
          left_dist = sampled_depth[j * max_steps + k] - min_depth[j * max_hits + cnt];
        } else {
          left_dist = (sampled_depth[j * max_steps + k] - sampled_depth[j * max_steps + k - 1]) / 2.0;
        }
      }
      previous_cnt = cnt;      
      depth = depth + (step_size - delta);  // go to next step
    }

    if (!done) {
      right_dist = min_depth[j * max_hits + previous_cnt] - sampled_depth[j * max_steps + max_steps - 1];
      sampled_dists[j * max_steps + max_steps - 1] = left_dist + right_dist;
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



/* backup 
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

  for (int j = index; j < num_rays; j += stride) {
    float depth = 0.0, delta = 0.0, left_dist = 0.0, right_dist = 0.0;
    bool done = false;
    int previous_cnt = -1;
    
    for (int l = 0; l < max_steps; ++l) {
      sampled_idx[j * max_steps + l] = -1;
    }

    for (int k = 0, cnt = 0; k < max_steps && cnt < max_hits; ++k) {

      if (pts_idx[j * max_hits + cnt] == -1) {
        done = true;
        break;
      }

      delta = step_size * uniform_noise[j * max_steps + k];   // stratified samples 
      depth = depth + delta;

      while (depth > (max_depth[j * max_hits + cnt] - min_depth[j * max_hits + cnt])) {
        depth -= (max_depth[j * max_hits + cnt] - min_depth[j * max_hits + cnt]);
        cnt += 1;
        
        if ((cnt == max_hits) || (pts_idx[j * max_hits + cnt] == -1)){
          done = true;
          break;
        }
      }

      sampled_idx[j * max_steps + k] = pts_idx[j * max_hits + cnt];
      sampled_depth[j * max_steps + k] = depth + min_depth[j * max_hits + cnt];
      
      // right distance for last step
      if (previous_cnt != -1) {
        if (cnt != previous_cnt) {
          right_dist = max_depth[j * max_hits + previous_cnt] - sampled_depth[j * max_steps + k - 1];
        } else {
          right_dist = (sampled_depth[j * max_steps + k] - sampled_depth[j * max_steps + k - 1]) / 2.0;
        }
        sampled_dists[j * max_steps + k - 1] = left_dist + right_dist;
      }
      if (done) break;

      // left distance for current step
      if (cnt != -1) {
        if (cnt != previous_cnt) {  // cross voxel boundary
          left_dist = sampled_depth[j * max_steps + k] - min_depth[j * max_hits + cnt];
        } else {
          left_dist = (sampled_depth[j * max_steps + k] - sampled_depth[j * max_steps + k - 1]) / 2.0;
        }
      }
      previous_cnt = cnt;      
      depth = depth + (step_size - delta);  // go to next step
    }

    if (!done) {
      right_dist = min_depth[j * max_hits + previous_cnt] - sampled_depth[j * max_steps + max_steps - 1];
      sampled_dists[j * max_steps + max_steps - 1] = left_dist + right_dist;
    }
  }
}

*/
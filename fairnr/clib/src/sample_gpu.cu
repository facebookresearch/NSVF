// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"  // required for float3 vector math


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
      if ((umax == max_hits) || (ucur == max_steps) || (pts_idx[H + umax] == -1)) {
        break;  // reach the maximum
      }
      if (umin < max_hits) {
        last_min_depth = min_depth[H + umin];
      } else {
        last_min_depth = 10000.0;
      }
      if (umax < max_hits) {
        last_max_depth = max_depth[H + umax];
      } else {
        last_max_depth = 10000.0;
      }
      if (ucur < max_steps) {
        curr_depth = min_depth[H] + (float(ucur) + uniform_noise[K + ucur]) * step_size;
      }
      
      if ((last_max_depth <= curr_depth) && (last_max_depth <= last_min_depth)) {
        sampled_depth[K + s] = last_max_depth;
        sampled_idx[K + s] = pts_idx[H + umax];
        umax++; s++; continue;
      }
      if ((curr_depth <= last_min_depth) && (curr_depth <= last_max_depth)) {
        sampled_depth[K + s] = curr_depth;
        sampled_idx[K + s] = pts_idx[H + umin - 1];
        ucur++; s++; continue;
      }
      if ((last_min_depth <= curr_depth) && (last_min_depth <= last_max_depth)) {
        sampled_depth[K + s] = last_min_depth;
        sampled_idx[K + s] = pts_idx[H + umin];
        umin++; s++; continue;
      }
    }

    float l_depth, r_depth;
    int step = 0;
    for (ucur = 0, umin = 0, umax = 0; ucur < max_steps - 1; ucur++) {
      if (sampled_idx[K + ucur + 1] == -1) break;
      l_depth = sampled_depth[K + ucur];
      r_depth = sampled_depth[K + ucur + 1];  
      sampled_depth[K + ucur] = (l_depth + r_depth) * .5;
      sampled_dists[K + ucur] = (r_depth - l_depth);
      if ((umin < max_hits) && (sampled_depth[K + ucur] >= min_depth[H + umin]) && (pts_idx[H + umin] > -1)) umin++;
      if ((umax < max_hits) && (sampled_depth[K + ucur] >= max_depth[H + umax]) && (pts_idx[H + umax] > -1)) umax++;
      if ((umax == max_hits) || (pts_idx[H + umax] == -1)) break;
      if ((umin - 1 == umax) && (sampled_dists[K + ucur] > 0)) {
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

__global__ void inverse_cdf_sampling_kernel(
    int b, int num_rays, 
    int max_hits,
    int max_steps,
    float fixed_step_size,
    const int *__restrict__ pts_idx,
    const float *__restrict__ min_depth,
    const float *__restrict__ max_depth,
    const float *__restrict__ uniform_noise,
    const float *__restrict__ probs,
    const float *__restrict__ steps,
    int *__restrict__ sampled_idx,
    float *__restrict__ sampled_depth,
    float *__restrict__ sampled_dists) {

    int batch_index = blockIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;

    pts_idx += batch_index * num_rays * max_hits;
    min_depth += batch_index * num_rays * max_hits;
    max_depth += batch_index * num_rays * max_hits;
    probs += batch_index * num_rays * max_hits;
    steps += batch_index * num_rays;

    uniform_noise += batch_index * num_rays * max_steps;
    sampled_idx += batch_index * num_rays * max_steps;
    sampled_depth += batch_index * num_rays * max_steps;
    sampled_dists += batch_index * num_rays * max_steps;

    // loop over all rays
    for (int j = index; j < num_rays; j += stride) {
        int H = j * max_hits, K = j * max_steps;
        int curr_bin = 0, s = 0;  // current index (bin)

        float curr_min_depth = min_depth[H];  // lower depth
        float curr_max_depth = max_depth[H];  // upper depth
        float curr_min_cdf = 0;
        float curr_max_cdf = probs[H];
        float step_size = 1.0 / steps[j];
        float z_low = curr_min_depth;        
        int total_steps = int(ceil(steps[j]));
        bool done = false;

        // optional use a fixed step size
        if (fixed_step_size > 0.0) step_size = fixed_step_size;

        // sample points 
        for (int curr_step = 0; curr_step < total_steps; curr_step++) {
            float curr_cdf = (float(curr_step) + uniform_noise[K + curr_step]) * step_size;
            while (curr_cdf > curr_max_cdf) {
                // first include max cdf
                sampled_idx[K + s] = pts_idx[H + curr_bin];
                sampled_dists[K + s] = (curr_max_depth - z_low);
                sampled_depth[K + s] = (curr_max_depth + z_low) * .5;

                // move to next cdf
                curr_bin++; 
                s++;
                if ((curr_bin >= max_hits) || (pts_idx[H + curr_bin] == -1)) {
                    done = true; break;
                }
                curr_min_depth = min_depth[H + curr_bin];
                curr_max_depth = max_depth[H + curr_bin];
                curr_min_cdf = curr_max_cdf;
                curr_max_cdf = curr_max_cdf + probs[H + curr_bin];
                z_low = curr_min_depth;
            }
            if (done) break;
            
            // if the sampled cdf is inside bin
            float u = (curr_cdf - curr_min_cdf) / (curr_max_cdf - curr_min_cdf);
            float z = curr_min_depth + u * (curr_max_depth - curr_min_depth);
            sampled_idx[K + s] = pts_idx[H + curr_bin];
            sampled_dists[K + s] = (z - z_low);
            sampled_depth[K + s] = (z + z_low) * .5;
            z_low = z; s++;
        }
        
        // if there are bins still remained
        while ((z_low < curr_max_depth) && (~done)) {
            sampled_idx[K + s] = pts_idx[H + curr_bin];
            sampled_dists[K + s] = (curr_max_depth - z_low);
            sampled_depth[K + s] = (curr_max_depth + z_low) * .5;
            curr_bin++; 
            s++;
            if ((curr_bin >= max_hits) || (pts_idx[curr_bin] == -1)) 
                break;
            
            curr_min_depth = min_depth[H + curr_bin];
            curr_max_depth = max_depth[H + curr_bin];
            z_low = curr_min_depth;
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

void inverse_cdf_sampling_kernel_wrapper(
    int b, int num_rays, int max_hits, int max_steps, float fixed_step_size,
    const int *pts_idx, const float *min_depth, const float *max_depth,
    const float *uniform_noise, const float *probs, const float *steps,
    int *sampled_idx, float *sampled_depth, float *sampled_dists) {
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    inverse_cdf_sampling_kernel<<<b, opt_n_threads(num_rays), 0, stream>>>(
        b, num_rays, max_hits, max_steps, fixed_step_size,
        pts_idx, min_depth, max_depth, uniform_noise, probs, steps, 
        sampled_idx, sampled_depth, sampled_dists);
    
    CUDA_CHECK_ERRORS();
}
  
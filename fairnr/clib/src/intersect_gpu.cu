// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "cutil_math.h"  // required for float3 vector math


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
        float depth_blur = sqrt(radius2 - r2);
        
        min_depth[j * n_max + cnt] = depth - depth_blur;
        max_depth[j * n_max + cnt] = depth + depth_blur;
        ++cnt;
      }
    }
  }
}


__device__ float2 RayAABBIntersection(
  const float3 &ori,
  const float3 &dir,
  const float3 &center,
  float half_voxel) {

  float f_low = 0;
  float f_high = 100000.;
  float f_dim_low, f_dim_high, temp, inv_ray_dir, start, aabb;

  for (int d = 0; d < 3; ++d) {  
    switch (d) {
      case 0:
        inv_ray_dir = __fdividef(1.0f, dir.x); start = ori.x; aabb = center.x; break;
      case 1:
        inv_ray_dir = __fdividef(1.0f, dir.y); start = ori.y; aabb = center.y; break;
      case 2:
        inv_ray_dir = __fdividef(1.0f, dir.z); start = ori.z; aabb = center.z; break;
    }
  
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
      return make_float2(-1.0f, -1.0f);
    }
  
    // Likewise if the low is less than the high.
    if (f_dim_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
      
    // Add the clip from this dimension to the previous results 
    f_low = (f_dim_low > f_low) ? f_dim_low : f_low;
    f_high = (f_dim_high < f_high) ? f_dim_high : f_high;
    
    if (f_low > f_high) {
      return make_float2(-1.0f, -1.0f);
    }
  }
  return make_float2(f_low, f_high);
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
      float2 depths = RayAABBIntersection(
        make_float3(ray_start[j * 3 + 0], ray_start[j * 3 + 1], ray_start[j * 3 + 2]),
        make_float3(ray_dir[j * 3 + 0], ray_dir[j * 3 + 1], ray_dir[j * 3 + 2]),
        make_float3(points[k * 3 + 0], points[k * 3 + 1], points[k * 3 + 2]),
        half_voxel);

      if (depths.x > -1.0f){
        idx[j * n_max + cnt] = k;
        min_depth[j * n_max + cnt] = depths.x;
        max_depth[j * n_max + cnt] = depths.y;
        ++cnt;
      }
    }
  }
}


__global__ void svo_intersect_point_kernel(
            int b, int n, int m, float voxelsize,
            int n_max,
            const float *__restrict__ ray_start,
            const float *__restrict__ ray_dir,
            const float *__restrict__ points,
            const int *__restrict__ children,
            int *__restrict__ idx,
            float *__restrict__ min_depth,
            float *__restrict__ max_depth) {
  /*
  TODO: this is an inefficient implementation of the 
        navie Ray -- Sparse Voxel Octree Intersection. 
        It can be further improved using:
        
        Revelles, Jorge, Carlos Urena, and Miguel Lastra. 
        "An efficient parametric algorithm for octree traversal." (2000).
  */
  int batch_index = blockIdx.x;
  points += batch_index * n * 3;
  children += batch_index * n * 9;
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
    int stack[256] = {-1};    // DFS, initialize the stack
    int ptr = 0, cnt = 0, k = -1;
    stack[ptr] = n - 1;       // ROOT node is always the last
    while (ptr > -1 && cnt < n_max) {
      assert((ptr < 256));

      // evaluate the current node
      k = stack[ptr];
      float2 depths = RayAABBIntersection(
        make_float3(ray_start[j * 3 + 0], ray_start[j * 3 + 1], ray_start[j * 3 + 2]),
        make_float3(ray_dir[j * 3 + 0], ray_dir[j * 3 + 1], ray_dir[j * 3 + 2]),
        make_float3(points[k * 3 + 0], points[k * 3 + 1], points[k * 3 + 2]),
        half_voxel * float(children[k * 9 + 8]));
      stack[ptr] = -1; ptr--;

      if (depths.x > -1.0f) { // ray did not miss the voxel
        // TODO: here it should be able to know which children is ok, further optimize the code
        if (children[k * 9 + 8] == 1) {  // this is a terminal node 
          idx[j * n_max + cnt] = k;
          min_depth[j * n_max + cnt] = depths.x;
          max_depth[j * n_max + cnt] = depths.y;
          ++cnt; continue;
        }

        for (int u = 0; u < 8; u++) {
          if (children[k * 9 + u] > -1) {
            ptr++; stack[ptr] = children[k * 9 + u]; // push child to the stack
          }
        }  
      }
    }
  }
}


__device__ float3 RayTriangleIntersection(
  const float3 &ori,
  const float3 &dir,
	const float3 &v0,
	const float3 &v1,
  const float3 &v2,
  float blur) {
  
  float3 v0v1 = v1 - v0;
  float3 v0v2 = v2 - v0;
  float3 v0O = ori - v0;
  float3 dir_crs_v0v2 = cross(dir, v0v2);
  
  float det = dot(v0v1, dir_crs_v0v2);
  det = __fdividef(1.0f, det);  // CUDA intrinsic function 
  
	float u = dot(v0O, dir_crs_v0v2) * det;
	if (u < 0.0f - blur || u > 1.0f + blur)
		return make_float3(-1.0f, 0.0f, 0.0f);

  float3 v0O_crs_v0v1 = cross(v0O, v0v1);
	float v = dot(dir, v0O_crs_v0v1) * det;
	if (v < 0.0f - blur || v > 1.0f + blur)
    return make_float3(-1.0f, 0.0f, 0.0f);
    
  if ((u + v) < 0.0f - blur || (u + v) > 1.0f + blur)
    return make_float3(-1.0f, 0.0f, 0.0f);

  float t = dot(v0v2, v0O_crs_v0v1) * det;
  return make_float3(t, u, v);
}


__global__ void triangle_intersect_point_kernel(
            int b, int n, int m, float cagesize,
            float blur, int n_max,
            const float *__restrict__ ray_start,
            const float *__restrict__ ray_dir,
            const float *__restrict__ face_points,
            int *__restrict__ idx,
            float *__restrict__ depth,
            float *__restrict__ uv) {
  
  int batch_index = blockIdx.x;
  face_points += batch_index * n * 9;
  ray_start += batch_index * m * 3;
  ray_dir += batch_index * m * 3;
  idx += batch_index * m * n_max;
  depth += batch_index * m * n_max * 3;
  uv += batch_index * m * n_max * 2;
    
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int j = index; j < m; j += stride) {
    // go over rays
    for (int l = 0; l < n_max; ++l) {
      idx[j * n_max + l] = -1;
    }

    int cnt = 0;
    for (int k = 0; k < n && cnt < n_max; ++k) {
      // go over triangles
      float3 tuv = RayTriangleIntersection(
        make_float3(ray_start[j * 3 + 0], ray_start[j * 3 + 1], ray_start[j * 3 + 2]),
        make_float3(ray_dir[j * 3 + 0], ray_dir[j * 3 + 1], ray_dir[j * 3 + 2]),
        make_float3(face_points[k * 9 + 0], face_points[k * 9 + 1], face_points[k * 9 + 2]),
        make_float3(face_points[k * 9 + 3], face_points[k * 9 + 4], face_points[k * 9 + 5]),
        make_float3(face_points[k * 9 + 6], face_points[k * 9 + 7], face_points[k * 9 + 8]),
        blur);

      if (tuv.x > 0) {
        int ki = k;
        float d = tuv.x, u = tuv.y, v = tuv.z;

        // sort
        for (int l = 0; l < cnt; l++) {
          if (d < depth[j * n_max * 3 + l * 3]) {
            swap(ki, idx[j * n_max + l]);
            swap(d, depth[j * n_max * 3 + l * 3]);
            swap(u, uv[j * n_max * 2 + l * 2]);
            swap(v, uv[j * n_max * 2 + l * 2 + 1]);
          }
        }
        idx[j * n_max + cnt] = ki;
        depth[j * n_max * 3 + cnt * 3] = d;
        uv[j * n_max * 2 + cnt * 2] = u;
        uv[j * n_max * 2 + cnt * 2 + 1] = v;
        cnt++;
      }
    }

    for (int l = 0; l < cnt; l++) {
      // compute min_depth
      if (l == 0) 
        depth[j * n_max * 3 + l * 3 + 1] = -cagesize;
      else
        depth[j * n_max * 3 + l * 3 + 1] = -fminf(cagesize, 
          .5 * (depth[j * n_max * 3 + l * 3] - depth[j * n_max * 3 + l * 3 - 3]));
      
      // compute max_depth
      if (l == cnt - 1)
        depth[j * n_max * 3 + l * 3 + 2] = cagesize;
      else
        depth[j * n_max * 3 + l * 3 + 2] = fminf(cagesize, 
          .5 * (depth[j * n_max * 3 + l * 3 + 3] - depth[j * n_max * 3 + l * 3]));
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


void svo_intersect_point_kernel_wrapper(
  int b, int n, int m, float voxelsize, int n_max,
  const float *ray_start, const float *ray_dir, const float *points, const int *children,
  int *idx, float *min_depth, float *max_depth) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  svo_intersect_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, voxelsize, n_max, ray_start, ray_dir, points, children, idx, min_depth, max_depth);
  
  CUDA_CHECK_ERRORS();
}


void triangle_intersect_point_kernel_wrapper(
  int b, int n, int m, float cagesize, float blur, int n_max,
  const float *ray_start, const float *ray_dir, const float *face_points,
  int *idx, float *depth, float *uv) {
  
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  triangle_intersect_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, cagesize, blur, n_max, ray_start, ray_dir, face_points, idx, depth, uv);
  
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
      if (umax == max_hits || ucur == max_steps || pts_idx[H + umax] == -1) {
        break;  // reach the maximum
      }
      if (umin < max_hits && pts_idx[H + umin] > -1) {
        last_min_depth = min_depth[H + umin];
      } else {
        last_min_depth = last_max_depth + 0.1;
      }
      if (umax < max_hits && pts_idx[H + umin] > -1) {
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
      if (sampled_idx[K + ucur + 1] == -1) break;
      l_depth = sampled_depth[K + ucur];
      r_depth = sampled_depth[K + ucur + 1];  
      sampled_depth[K + ucur] = (l_depth + r_depth) * .5;
      sampled_dists[K + ucur] = (r_depth - l_depth);
      if (umin < max_hits && sampled_depth[K + ucur] >= min_depth[H + umin] && pts_idx[H + umin] > -1) umin++;
      if (umax < max_hits && sampled_depth[K + ucur] >= max_depth[H + umax] && pts_idx[H + umax] > -1) umax++;
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


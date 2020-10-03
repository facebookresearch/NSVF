// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "intersect.h"
#include "utils.h"
#include <utility> 

void ball_intersect_point_kernel_wrapper(
  int b, int n, int m, float radius, int n_max,
  const float *ray_start, const float *ray_dir, const float *points,
  int *idx, float *min_depth, float *max_depth);

std::tuple< at::Tensor, at::Tensor, at::Tensor > ball_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor points, 
               const float radius, const int n_max){
  CHECK_CONTIGUOUS(ray_start);
  CHECK_CONTIGUOUS(ray_dir);
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(ray_start);
  CHECK_IS_FLOAT(ray_dir);
  CHECK_IS_FLOAT(points);
  CHECK_CUDA(ray_start);
  CHECK_CUDA(ray_dir);
  CHECK_CUDA(points);

  at::Tensor idx =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                    at::device(ray_start.device()).dtype(at::ScalarType::Int));
  at::Tensor min_depth =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                    at::device(ray_start.device()).dtype(at::ScalarType::Float));
  at::Tensor max_depth =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                    at::device(ray_start.device()).dtype(at::ScalarType::Float));
  ball_intersect_point_kernel_wrapper(points.size(0), points.size(1), ray_start.size(1),
                                      radius, n_max,
                                      ray_start.data<float>(), ray_dir.data<float>(), points.data<float>(),
                                      idx.data<int>(), min_depth.data<float>(), max_depth.data<float>());
  return std::make_tuple(idx, min_depth, max_depth);
}


void aabb_intersect_point_kernel_wrapper(
  int b, int n, int m, float voxelsize, int n_max,
  const float *ray_start, const float *ray_dir, const float *points,
  int *idx, float *min_depth, float *max_depth);

std::tuple< at::Tensor, at::Tensor, at::Tensor > aabb_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor points, 
               const float voxelsize, const int n_max){
  CHECK_CONTIGUOUS(ray_start);
  CHECK_CONTIGUOUS(ray_dir);
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(ray_start);
  CHECK_IS_FLOAT(ray_dir);
  CHECK_IS_FLOAT(points);
  CHECK_CUDA(ray_start);
  CHECK_CUDA(ray_dir);
  CHECK_CUDA(points);

  at::Tensor idx =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                    at::device(ray_start.device()).dtype(at::ScalarType::Int));
  at::Tensor min_depth =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                    at::device(ray_start.device()).dtype(at::ScalarType::Float));
  at::Tensor max_depth =
      torch::zeros({ray_start.size(0), ray_start.size(1), n_max},
                    at::device(ray_start.device()).dtype(at::ScalarType::Float));
  aabb_intersect_point_kernel_wrapper(points.size(0), points.size(1), ray_start.size(1),
                                      voxelsize, n_max,
                                      ray_start.data<float>(), ray_dir.data<float>(), points.data<float>(),
                                      idx.data<int>(), min_depth.data<float>(), max_depth.data<float>());
  return std::make_tuple(idx, min_depth, max_depth);
}


void uniform_ray_sampling_kernel_wrapper(
  int b, int num_rays, int max_hits, int max_steps, float step_size,
  const int *pts_idx, const float *min_depth, const float *max_depth, const float *uniform_noise,
  int *sampled_idx, float *sampled_depth, float *sampled_dists);


std::tuple< at::Tensor, at::Tensor, at::Tensor> uniform_ray_sampling(
  at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
  const float step_size, const int max_steps){

  CHECK_CONTIGUOUS(pts_idx);
  CHECK_CONTIGUOUS(min_depth);
  CHECK_CONTIGUOUS(max_depth);
  CHECK_CONTIGUOUS(uniform_noise);
  CHECK_IS_FLOAT(min_depth);
  CHECK_IS_FLOAT(max_depth);
  CHECK_IS_FLOAT(uniform_noise);
  CHECK_CUDA(pts_idx);
  CHECK_CUDA(min_depth);
  CHECK_CUDA(max_depth);
  CHECK_CUDA(uniform_noise);

  at::Tensor sampled_idx =
      -torch::ones({pts_idx.size(0), pts_idx.size(1), max_steps},
                    at::device(pts_idx.device()).dtype(at::ScalarType::Int));
  at::Tensor sampled_depth =
      torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                    at::device(min_depth.device()).dtype(at::ScalarType::Float));
  at::Tensor sampled_dists =
      torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                    at::device(min_depth.device()).dtype(at::ScalarType::Float));
  uniform_ray_sampling_kernel_wrapper(min_depth.size(0), min_depth.size(1), min_depth.size(2), sampled_depth.size(2),
                                      step_size,
                                      pts_idx.data<int>(), min_depth.data<float>(), max_depth.data<float>(),
                                      uniform_noise.data<float>(), sampled_idx.data<int>(), 
                                      sampled_depth.data<float>(), sampled_dists.data<float>());
  return std::make_tuple(sampled_idx, sampled_depth, sampled_dists);
}
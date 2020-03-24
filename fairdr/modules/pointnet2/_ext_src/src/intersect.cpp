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
// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "sample.h"
#include "utils.h"
#include <utility> 


void uniform_ray_sampling_kernel_wrapper(
  int b, int num_rays, int max_hits, int max_steps, float step_size,
  const int *pts_idx, const float *min_depth, const float *max_depth, const float *uniform_noise,
  int *sampled_idx, float *sampled_depth, float *sampled_dists);

void inverse_cdf_sampling_kernel_wrapper(
  int b, int num_rays, int max_hits, int max_steps, float fixed_step_size,
  const int *pts_idx, const float *min_depth, const float *max_depth,
  const float *uniform_noise, const float *probs, const float *steps,
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
  CHECK_IS_INT(pts_idx);
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
                                      pts_idx.data_ptr <int>(), min_depth.data_ptr <float>(), max_depth.data_ptr <float>(),
                                      uniform_noise.data_ptr <float>(), sampled_idx.data_ptr <int>(), 
                                      sampled_depth.data_ptr <float>(), sampled_dists.data_ptr <float>());
  return std::make_tuple(sampled_idx, sampled_depth, sampled_dists);
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> inverse_cdf_sampling(
    at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
    at::Tensor probs, at::Tensor steps, float fixed_step_size) {
  
  CHECK_CONTIGUOUS(pts_idx);
  CHECK_CONTIGUOUS(min_depth);
  CHECK_CONTIGUOUS(max_depth);
  CHECK_CONTIGUOUS(probs);
  CHECK_CONTIGUOUS(steps);
  CHECK_CONTIGUOUS(uniform_noise);
  CHECK_IS_FLOAT(min_depth);
  CHECK_IS_FLOAT(max_depth);
  CHECK_IS_FLOAT(uniform_noise);
  CHECK_IS_FLOAT(probs);
  CHECK_IS_FLOAT(steps);
  CHECK_IS_INT(pts_idx);
  CHECK_CUDA(pts_idx);
  CHECK_CUDA(min_depth);
  CHECK_CUDA(max_depth);
  CHECK_CUDA(uniform_noise);
  CHECK_CUDA(probs);
  CHECK_CUDA(steps);

  int max_steps = uniform_noise.size(-1);
  at::Tensor sampled_idx =
      -torch::ones({pts_idx.size(0), pts_idx.size(1), max_steps},
                    at::device(pts_idx.device()).dtype(at::ScalarType::Int));
  at::Tensor sampled_depth =
      torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                    at::device(min_depth.device()).dtype(at::ScalarType::Float));
  at::Tensor sampled_dists =
      torch::zeros({min_depth.size(0), min_depth.size(1), max_steps},
                    at::device(min_depth.device()).dtype(at::ScalarType::Float));
  inverse_cdf_sampling_kernel_wrapper(min_depth.size(0), min_depth.size(1), min_depth.size(2), sampled_depth.size(2), fixed_step_size,
                                      pts_idx.data_ptr <int>(), min_depth.data_ptr <float>(), max_depth.data_ptr <float>(),
                                      uniform_noise.data_ptr <float>(), probs.data_ptr <float>(), steps.data_ptr <float>(),
                                      sampled_idx.data_ptr <int>(), sampled_depth.data_ptr <float>(), sampled_dists.data_ptr <float>());
  return std::make_tuple(sampled_idx, sampled_depth, sampled_dists);
}
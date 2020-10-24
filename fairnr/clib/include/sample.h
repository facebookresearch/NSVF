// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>
#include <utility>

            
std::tuple<at::Tensor, at::Tensor, at::Tensor> uniform_ray_sampling(
    at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
    const float step_size, const int max_steps);
std::tuple<at::Tensor, at::Tensor, at::Tensor> inverse_cdf_sampling(
    at::Tensor pts_idx, at::Tensor min_depth, at::Tensor max_depth, at::Tensor uniform_noise,
    at::Tensor probs, at::Tensor steps, float fixed_step_size);
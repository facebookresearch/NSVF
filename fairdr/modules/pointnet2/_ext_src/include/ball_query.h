// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>
#include <utility>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius, const int nsample);
std::pair<at::Tensor, at::Tensor> ball_nearest(at::Tensor new_xyz, at::Tensor xyz, const float radius);
// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>
#include <utility>

std::tuple<at::Tensor, at::Tensor, at::Tensor> ball_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor points, 
               const float radius, const int n_max);
std::tuple<at::Tensor, at::Tensor, at::Tensor> aabb_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor points, 
               const float voxelsize, const int n_max);
std::tuple<at::Tensor, at::Tensor, at::Tensor> svo_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor points, at::Tensor children,
               const float voxelsize, const int n_max);
std::tuple< at::Tensor, at::Tensor, at::Tensor > triangle_intersect(at::Tensor ray_start, at::Tensor ray_dir, at::Tensor face_points, 
               const float cagesize, const float blur, const int n_max);              

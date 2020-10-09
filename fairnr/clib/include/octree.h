// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>
#include <utility>

std::tuple<at::Tensor, at::Tensor> build_octree(at::Tensor center, at::Tensor points, int depth);
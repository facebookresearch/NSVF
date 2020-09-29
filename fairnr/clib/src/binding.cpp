// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "intersect.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ball_intersect", &ball_intersect);
  m.def("aabb_intersect", &aabb_intersect);
  m.def("uniform_ray_sampling", &uniform_ray_sampling);
}
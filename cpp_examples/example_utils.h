/* Copyright 2020 Stanford, Tsinghua
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _CPP_EXAMPLES_EXAMPLE_UTILS_H_
#define _CPP_EXAMPLES_EXAMPLE_UTILS_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>
#include "taso/ops.h"

using namespace taso;

DATATYPE* new_random_data(size_t size) {
  // Random generator.
  static std::random_device r;
  static std::default_random_engine e(r());
  static std::uniform_real_distribution<DATATYPE> dist;
  auto gen = [&]() { return dist(e); };

  auto data = new DATATYPE[size];
  std::generate(data, data + size, gen);
  return data;
}

size_t dims2size(const std::vector<int>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

inline TensorHandle new_input(Graph* graph, const std::vector<int>& dims) {
  return graph->new_input(dims.size(), dims.data());
}

inline TensorHandle new_weight(Graph* graph, const std::vector<int>& dims, const DATATYPE* data) {
  return graph->new_weight(dims.size(), dims.data(), data);
}

inline TensorHandle new_random_weight(Graph* graph, const std::vector<int>& dims) {
  return new_weight(graph, dims, new_random_data(dims2size(dims)));
}

#endif

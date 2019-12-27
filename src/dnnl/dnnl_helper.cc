/* Copyright 2019 Stanford
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

#include "taso/dnnl_helper.h"
using namespace taso;

void assign_kernel(DATATYPE* ptr, int size, DATATYPE value) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) ptr[i] = value;
}

void copy_kernel(DATATYPE* dst, const DATATYPE* src, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i++) dst[i] = src[i];
}

dnnl::primitive_attr get_activation_attr(ActiMode activation) {
  dnnl::algorithm ops_algo = dnnl::algorithm::eltwise_relu;  // relu as default
  const float ops_scale = 1.0f;
  float ops_alpha = 0.0f;  // relu negative slope
  float ops_beta = 0.0f;
  switch (activation) {
    case AC_MODE_SIGMOID:
      ops_algo = dnnl::algorithm::eltwise_logistic;
      // alpha and beta ignored.
      break;
    case AC_MODE_TANH:
      ops_algo = dnnl::algorithm::eltwise_tanh;
      // alpha and beta ignored.
      break;
    case AC_MODE_RELU:
    default:
      break;
  }
  dnnl::post_ops ops;
  ops.append_eltwise(ops_scale, ops_algo, ops_alpha, ops_beta);
  dnnl::primitive_attr attr;
  attr.set_post_ops(ops);
  return attr;
}

dnnl::memory::desc get_memory_desc(const Tensor& t, int numDim) {
  if (numDim <= 0) numDim = t.numDim;
  assert(numDim >= t.numDim);
  dnnl::memory::dims size;
  dnnl::memory::dims stride;
  // right-align the dimensions
  for (int i = 0; i < numDim - t.numDim; i++) {
    size.push_back(1);
    stride.push_back(t.stride[0]);
  }
  for (int i = 0; i < t.numDim; i++) {
    size.push_back(t.dim[i]);
    stride.push_back(t.stride[i]);
  }
  assert(size.size() == (size_t)numDim);
  assert(stride.size() == (size_t)numDim);
  return dnnl::memory::desc(size, DNNL_DEF_DTYPE, stride);
}


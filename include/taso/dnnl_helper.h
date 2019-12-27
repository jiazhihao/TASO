/* Copyright 2019 Stanford, Tsinghua
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

#ifndef _DNNL_HELPER_H_
#define _DNNL_HELPER_H_

#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include "taso/ops.h"
#include "dnnl.hpp"

#define _STR(x) #x
#define STR(x) _STR(x)

#define _ERROR_HEAD \
  std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] "

#define CHECK_EQ(expect, actual) if ((expect) != (actual)) {            \
  _ERROR_HEAD << "value != " << STR(expect) << std::endl;               \
  exit(1);                                                              \
}

#define CHECK_NE(notExpect, actual) if ((notExpect) == (actual)) {      \
  _ERROR_HEAD << "value == " << STR(notExpect) << std::endl;            \
  exit(1);                                                              \
}

inline uint64_t microsecond_timer() {
  auto t = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
}

static constexpr dnnl::memory::data_type DNNL_DEF_DTYPE = dnnl::memory::data_type::f32;
static constexpr dnnl::memory::format_tag DNNL_FMT_ANY = dnnl::memory::format_tag::any;

void assign_kernel(taso::DATATYPE* ptr, int size, taso::DATATYPE value);
void copy_kernel(taso::DATATYPE* dst, const taso::DATATYPE* src, int size);

dnnl::primitive_attr get_activation_attr(taso::ActiMode activation);

// if numDim is given to support broadcast, it must be no less than the tensor dimension.
dnnl::memory::desc get_memory_desc(const taso::Tensor& t, int numDim = 0);

#endif

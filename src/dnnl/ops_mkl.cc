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

#include "taso/ops.h"
#include "taso/dnnl_helper.h"
using namespace taso;

Model::Model()
: isTraining(false), print_cost(false)
{
  global_unique_id = 100;
  workSpaceSize = WORK_SPACE_SIZE;
  eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
  strm = dnnl::stream(eng);
  CHECK_NE(nullptr, workSpace = (DATATYPE*)malloc(workSpaceSize));
  // allocate tensors for measuring performance
  CHECK_NE(nullptr, inputPtr = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, biasPtr = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, outputPtr = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, filterPtr = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  // create tensors for batch norm
  CHECK_NE(nullptr, scalePtr = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, runningMean = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, runningVar = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, saveMean = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
  CHECK_NE(nullptr, saveVar = (DATATYPE*)malloc(MAX_TENSOR_SIZE));
}

float Model::measure_oplist_runtime(const std::vector<OpBase*>& opBaseList) {
  const int num_runs = 100;
  // warmup
  for (int times = 0; times < num_runs; times++)
    for (size_t i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
  // measure runtime
  auto beg = microsecond_timer();
  for (int times = 0; times < num_runs; times++) {
    for (size_t i = 0; i < opBaseList.size(); i++)
      opBaseList[i]->forward();
  }
  auto end = microsecond_timer();
  return (end - beg) / 1.e3 / num_runs;
}

void* Model::allocate_memory(size_t size, const DATATYPE* data_initial) {
  void* ptr;
  if (size == 0) {
    // Note: Special value for zero-sized tensor
    ptr = (void*) 0x1;
  } else {
    CHECK_NE(nullptr, ptr = malloc(size));
  }
  if (data_initial != NULL) {
    memcpy(ptr, data_initial, size);
  }
  return ptr;
}

bool Model::copy_memory(DATATYPE* dst, const DATATYPE* src, size_t size) {
  memcpy(dst, src, size);
  return true;
}

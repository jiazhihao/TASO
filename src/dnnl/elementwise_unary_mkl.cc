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

#include "taso/ops.h"
#include "taso/dnnl_helper.h"
#include <cmath>
using namespace taso;
using namespace dnnl;

void unary_kernel(int volume, OpType type, const DATATYPE* x, DATATYPE* y) {
  switch (type) {
    case OP_CEIL:
      {
#pragma omp parallel for
        for (int i = 0; i < volume; i++) y[i] = std::ceil(x[i]);
        break;
      }
    case OP_ROUND:
      {
#pragma omp parallel for
        for (int i = 0; i < volume; i++) y[i] = std::round(x[i]);
        break;
      }
    case OP_LOGICAL_NOT:
      {
#pragma omp parallel for
        for (int i = 0; i < volume; i++) y[i] = !x[i];
        break;
      }
    case OP_LOG:
      {
#pragma omp parallel for
        for (int i = 0; i < volume; i++) y[i] = std::log(x[i]);
        break;
      }
    default:
      assert(false);
  }
}

bool ElementWiseUnary::use_kernel(void) const
{
  switch (type) {
    case OP_EXP:
    case OP_SQRT:
      return true;
    default:
      return false;
  }
}

static void create_net(ElementWiseUnary* unary, DNNLNet& net, engine& eng, stream& strm,
    memory& inputMem, memory& outputMem,
    void* inputPtr, void* outputPtr) {
  // dimensions.
  assert(unary->inputs[0].volume() == unary->outputs[0].volume());
  if (unary->use_kernel()) {
    // dnnl::eltwise requires the same layout between input and output.
    assert(unary->outputs[0].has_same_shape_stride_split(unary->inputs[0]));
    // data descriptors.
    auto memDesc = get_memory_desc(unary->outputs[0]);
    // data memories.
    inputMem = memory(memDesc, eng, inputPtr);
    outputMem = memory(memDesc, eng, outputPtr);
    // operator primitives.
    algorithm ops_algo;
    float ops_alpha = 0.0f;
    float ops_beta = 0.0f;
    switch (unary->type) {
      case OP_EXP:
        ops_algo = algorithm::eltwise_exp;
        break;
      case OP_SQRT:
        ops_algo = algorithm::eltwise_sqrt;
        break;
      default:
        assert(false);
    }
    auto unaryOpDesc = eltwise_forward::desc(
        prop_kind::forward_inference, ops_algo, memDesc, ops_alpha, ops_beta);
    auto unaryPrimDesc = eltwise_forward::primitive_desc(unaryOpDesc, eng);
    // create primitives and connect.
    net.clear();
    net.push_back({eltwise_forward(unaryPrimDesc),
        {{DNNL_ARG_SRC, inputMem},
        {DNNL_ARG_DST, outputMem}}});
  } else {
    // No preprocessing for our customized kernel
  }
}

void ElementWiseUnary::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  // create primitives.
  memory inputMem, outputMem;
  create_net(this, net, model->eng, model->strm,
      inputMem, outputMem,
      inputs[0].data_ptr, outputs[0].data_ptr);
}

void ElementWiseUnary::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void ElementWiseUnary::forward(bool block)
{
  if (use_kernel()) {
    for (auto& p : net) p.first.execute(model->strm, p.second);
    if (block) model->strm.wait();
  } else {
    unary_kernel(outputs[0].volume(), type,
        (DATATYPE*)inputs[0].data_ptr,
        (DATATYPE*)outputs[0].data_ptr);
  }
}

void Model::measure_elementwise_unary_cost(ElementWiseUnary* unary)
{
  memory inputMem, outputMem;
  create_net(unary, net, eng, strm,
      inputMem, outputMem,
      inputPtr, outputPtr);

  // measure.
  uint64_t beg = 0;
  if (unary->use_kernel()) {
    assert(inputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
    assert(outputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);

    for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
      if (i == WARMUP_TIMES) {
        beg = microsecond_timer();
      }
      for (auto& p : net) {
        p.first.execute(strm, p.second);
      }
      strm.wait();
    }
  } else {
    for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
      if (i == WARMUP_TIMES) {
        beg = microsecond_timer();
      }
      unary_kernel(unary->outputs[0].volume(), unary->type, inputPtr, outputPtr);
    }
  }
  auto end = microsecond_timer();

  unary->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[ElementWiseUnary]: type(%d) cost(%.4lf)\n",
           unary->type, unary->runtime);
}


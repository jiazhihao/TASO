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
using namespace dnnl;

static void create_net(Pool2D* pool, DNNLNet& net, engine& eng, stream& strm,
    memory& inputMem, memory& outputMem,
    void* inputPtr, void* outputPtr) {
  // dimensions.
  int padH, padW;
  pool->get_padding(&padH, &padW);
  // data sizes.
  const memory::dims kernelSize = { pool->kernelH, pool->kernelW };
  const memory::dims strideSize = { pool->strideH, pool->strideW };
  const memory::dims paddingSize = { padH, padW };
  // data descriptors.
  auto inputMemDesc = get_memory_desc(pool->inputs[0]);
  auto outputMemDesc = get_memory_desc(pool->outputs[0]);
  // data memories.
  inputMem = memory(inputMemDesc, eng, inputPtr);
  outputMem = memory(outputMemDesc, eng, outputPtr);
  // operator primitives.
  algorithm mode;
  switch (pool->type) {
    case OP_POOL2D_MAX:
      mode = algorithm::pooling_max;
      break;
    case OP_POOL2D_AVG:
      mode = algorithm::pooling_avg_exclude_padding;
      break;
    default:
      assert(false);
  }
  auto poolOpDesc = pooling_forward::desc(
      prop_kind::forward_inference, mode,
      inputMemDesc, outputMemDesc,
      strideSize, kernelSize, paddingSize, paddingSize);
  auto poolPrimDesc = pooling_forward::primitive_desc(poolOpDesc, eng);
  if (pool->activation != AC_MODE_NONE) {
    auto poolAttr = get_activation_attr(pool->activation);
    poolPrimDesc = pooling_forward::primitive_desc(poolOpDesc, poolAttr, eng);
  }
  // create primitives and connect.
  net.clear();
  assert(poolPrimDesc.dst_desc() == outputMem.get_desc());
  net.push_back({pooling_forward(poolPrimDesc),
      {{DNNL_ARG_SRC, inputMem},
      {DNNL_ARG_DST, outputMem}}});
}

void Pool2D::map(void)
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

void Pool2D::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void Pool2D::forward(bool block)
{
  for (auto& p : net) p.first.execute(model->strm, p.second);
  if (block) model->strm.wait();
}

void Model::measure_pool2d_cost(Pool2D* pool)
{
  memory inputMem, outputMem;
  create_net(pool, net, eng, strm,
      inputMem, outputMem,
      inputPtr, outputPtr);
  int padH, padW;
  pool->get_padding(&padH, &padW);

  assert(inputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(outputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);

  // measure.
  uint64_t beg = 0;
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES) {
      beg = microsecond_timer();
    }
    for (auto& p : net) {
      p.first.execute(strm, p.second);
    }
    strm.wait();
  }
  auto end = microsecond_timer();

  pool->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Pool2D]: i(%d %d %d %d) k(%d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
           pool->inputs[0].dim[0], pool->inputs[0].dim[1],
           pool->inputs[0].dim[2], pool->inputs[0].dim[3],
           pool->kernelH, pool->kernelW,
           pool->strideH, pool->strideW, padH, padW, pool->runtime);
}


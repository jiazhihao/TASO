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

static void create_net(Activation* act, DNNLNet& net, engine& eng, stream& strm,
    memory& inputMem, memory& outputMem,
    void* inputPtr, void* outputPtr) {
  // dimensions.
  assert(act->inputs[0].volume() == act->outputs[0].volume());
  // dnnl::eltwise requires the same layout between input and output.
  assert(act->outputs[0].has_same_shape_stride_split(act->inputs[0]));
  // data descriptors.
  auto memDesc = get_memory_desc(act->outputs[0]);
  // data memories.
  inputMem = memory(memDesc, eng, inputPtr);
  outputMem = memory(memDesc, eng, outputPtr);
  // operator primitives.
  algorithm ops_algo = algorithm::eltwise_relu;  // relu as default
  float ops_alpha = 0.0f;  // relu negative slope
  float ops_beta = 0.0f;
  switch (act->type) {
    case OP_RELU:
    case OP_LEAKYRELU:
      break;
    case OP_SIGMOID:
      ops_algo = dnnl::algorithm::eltwise_logistic;
      // alpha and beta ignored.
      break;
    case OP_TANH:
      ops_algo = dnnl::algorithm::eltwise_tanh;
      // alpha and beta ignored.
      break;
    default:
      assert(false);
  }
  auto actOpDesc = eltwise_forward::desc(
      prop_kind::forward_inference, ops_algo, memDesc, ops_alpha, ops_beta);
  auto actPrimDesc = eltwise_forward::primitive_desc(actOpDesc, eng);
  // create primitives and connect.
  net.clear();
  net.push_back({eltwise_forward(actPrimDesc),
      {{DNNL_ARG_SRC, inputMem},
      {DNNL_ARG_DST, outputMem}}});
}

void Activation::map(void)
{
  // allocate tensors
  if (!inPlace) {
    size_t outputSize = sizeof(DATATYPE) * inputs[0].volume();
    CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  } else {
    outputs[0].data_ptr = inputs[0].data_ptr;
  }
  // create primitives.
  memory inputMem, outputMem;
  create_net(this, net, model->eng, model->strm,
      inputMem, outputMem,
      inputs[0].data_ptr, outputs[0].data_ptr);
}

void Activation::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  if (!inPlace) {
    free(outputs[0].data_ptr);
    outputs[0].data_ptr = nullptr;
  }
}

void Activation::forward(bool block)
{
  for (auto& p : net) p.first.execute(model->strm, p.second);
  if (block) model->strm.wait();
}

void Model::measure_activation_cost(Activation* act)
{
  memory inputMem, outputMem;
  create_net(act, net, eng, strm,
      inputMem, outputMem,
      inputPtr, act->inPlace ? inputPtr : outputPtr);

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

  act->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Activation]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
           act->inputs[0].dim[0], act->inputs[0].dim[1], act->inputs[0].dim[2],
           act->inputs[0].dim[3], act->type, act->runtime);
}


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
using namespace taso;
using namespace dnnl;

static void create_net(Conv2D* conv, DNNLNet& net, engine& eng, stream& strm,
    memory& inputMem, memory& outputMem, memory& filterMem, memory& biasMem,
    void* inputPtr, void* outputPtr, void* filterPtr, void* biasPtr) {
  // dimensions.
  int inputC = conv->inputs[0].dim[1];
  int outputC = conv->outputs[0].dim[1];
  assert(outputC == conv->inputs[1].dim[0]);
  int kernelH = conv->inputs[1].dim[2];
  int kernelW = conv->inputs[1].dim[3];
  int groups = inputC / conv->inputs[1].dim[1];
  assert(conv->inputs[1].default_layout());
  int padH, padW;
  conv->get_padding(&padH, &padW);
  // data sizes.
  const memory::dims inputSize = memory::dims(conv->inputs[0].dim, conv->inputs[0].dim + conv->inputs[0].numDim);
  const memory::dims outputSize = memory::dims(conv->outputs[0].dim, conv->outputs[0].dim + conv->outputs[0].numDim);
  const memory::dims filterSize = { groups, outputC / groups, inputC / groups, kernelH, kernelW };
  const memory::dims biasSize = { outputC };
  const memory::dims strideSize = { conv->strideH, conv->strideW };
  const memory::dims paddingSize = { padH, padW };
  // data descriptors.
  auto inputMemDesc = get_memory_desc(conv->inputs[0]);
  auto outputMemDesc = get_memory_desc(conv->outputs[0]);
  auto filterMemDesc = memory::desc(filterSize, DNNL_DEF_DTYPE, memory::format_tag::goihw);
  auto biasMemDesc = memory::desc(biasSize, DNNL_DEF_DTYPE, memory::format_tag::x);
  // data memories.
  inputMem = memory(inputMemDesc, eng, inputPtr);
  outputMem = memory(outputMemDesc, eng, outputPtr);
  filterMem = memory(filterMemDesc, eng, filterPtr);
  biasMem = memory(biasMemDesc, eng, biasPtr);
  // operator primitives.
  auto convInputMemDesc = memory::desc(inputSize, DNNL_DEF_DTYPE, DNNL_FMT_ANY);
  auto convOutputMemDesc = memory::desc(outputSize, DNNL_DEF_DTYPE, DNNL_FMT_ANY);
  auto convFilterMemDesc = memory::desc(filterSize, DNNL_DEF_DTYPE, DNNL_FMT_ANY);
  auto convBiasMemDesc = memory::desc(biasSize, DNNL_DEF_DTYPE, DNNL_FMT_ANY);
  auto convOpDesc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct,
      convInputMemDesc, convFilterMemDesc, convBiasMemDesc, convOutputMemDesc,
      strideSize, paddingSize, paddingSize);
  auto convPrimDesc = convolution_forward::primitive_desc(convOpDesc, eng);
  if (conv->activation != AC_MODE_NONE) {
    auto convAttr = get_activation_attr(conv->activation);
    convPrimDesc = convolution_forward::primitive_desc(convOpDesc, convAttr, eng);
  }
  // create primitives and connect.
  net.clear();
  auto convInputMem = inputMem;
  auto convOutputMem = outputMem;
  auto convFilterMem = filterMem;
  auto convBiasMem = biasMem;
  if (convPrimDesc.src_desc() != inputMem.get_desc()) {
    convInputMem = memory(convPrimDesc.src_desc(), eng);
    net.push_back({reorder(inputMem, convInputMem),
        {{DNNL_ARG_FROM, inputMem}, {DNNL_ARG_TO, convInputMem}}});
  }
  if (convPrimDesc.weights_desc() != filterMem.get_desc()) {
    convFilterMem = memory(convPrimDesc.weights_desc(), eng);
    reorder(filterMem, convFilterMem).execute(strm, filterMem, convFilterMem);
    strm.wait();
  }
  assert(convPrimDesc.bias_desc() == biasMem.get_desc());
  net.push_back({convolution_forward(convPrimDesc),
      {{DNNL_ARG_SRC, convInputMem},
      {DNNL_ARG_WEIGHTS, convFilterMem},
      {DNNL_ARG_BIAS, convBiasMem},
      {DNNL_ARG_DST, convOutputMem}}});
  if (convPrimDesc.dst_desc() != outputMem.get_desc()) {
    convOutputMem = memory(convPrimDesc.dst_desc(), eng);
    net.push_back({reorder(convOutputMem, outputMem),
        {{DNNL_ARG_FROM, convOutputMem}, {DNNL_ARG_TO, outputMem}}});
  }
}

void Conv2D::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  CHECK_NE(nullptr, biasPtr = malloc(outputs[0].dim[1]));
  // create primitives.
  memory inputMem, outputMem, filterMem, biasMem;
  create_net(this, net, model->eng, model->strm,
      inputMem, outputMem, filterMem, biasMem,
      inputs[0].data_ptr, outputs[0].data_ptr, inputs[1].data_ptr, biasPtr);
}

void Conv2D::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  free(biasPtr);
  outputs[0].data_ptr = nullptr;
  biasPtr = nullptr;
}

void Conv2D::forward(bool block)
{
  for (auto& p : net) p.first.execute(model->strm, p.second);
  if (block) model->strm.wait();
}

void Model::measure_conv2d_cost(Conv2D* conv)
{
  memory inputMem, outputMem, filterMem, biasMem;
  create_net(conv, net, eng, strm,
      inputMem, outputMem, filterMem, biasMem,
      inputPtr, outputPtr, filterPtr, biasPtr);
  int padH, padW;
  conv->get_padding(&padH, &padW);

  assert(inputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(outputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(filterMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(biasMem.get_desc().get_size() <= MAX_TENSOR_SIZE);

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

  conv->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Conv2D]: i(%d %d %d %d) w(%d %d %d %d) s(%d %d) p(%d %d) cost(%.4lf)\n",
           conv->inputs[0].dim[0], conv->inputs[0].dim[1], conv->inputs[0].dim[2], conv->inputs[0].dim[3],
           conv->inputs[1].dim[0], conv->inputs[1].dim[1], conv->inputs[1].dim[2], conv->inputs[1].dim[3],
           conv->strideH, conv->strideW, padH, padW, conv->runtime);
}


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

static void create_net(BatchNorm* bn, DNNLNet& net, engine& eng, stream& strm,
    memory& inputMem, memory& outputMem, memory& meanMem, memory& varMem, memory& scaleShiftMem,
    void* inputPtr, void* outputPtr, void* meanPtr, void* varPtr, void* biasPtr,
    bool isTraining) {
  const float eps = epsilon;
  // dimensions.
  int inputC = bn->inputs[0].dim[1];
  // data sizes.
  const memory::dims statSize = { inputC };
  const memory::dims scaleShiftSize = { 2, inputC };
  // data descriptors.
  auto dataMemDesc = get_memory_desc(bn->inputs[0]);
  auto statMemDesc = memory::desc(statSize, DNNL_DEF_DTYPE, memory::format_tag::x);
  auto scaleShiftMemDesc = memory::desc(scaleShiftSize, DNNL_DEF_DTYPE, memory::format_tag::nc);
  // data memories.
  inputMem = memory(dataMemDesc, eng, inputPtr);
  outputMem = memory(dataMemDesc, eng, outputPtr);
  scaleShiftMem = memory(scaleShiftMemDesc, eng, biasPtr);
  meanMem = memory(statMemDesc, eng, meanPtr);
  varMem = memory(statMemDesc, eng, varPtr);
  // operator primitives.
  normalization_flags flags = normalization_flags::use_scale_shift | normalization_flags::use_global_stats;
  prop_kind prop = prop_kind::forward_inference;
#ifdef DO_TRAINING
  if (isTraining) {
    flags = normalization_flags::use_scale_shift;
    prop = prop_kind::forward_training;
    std::fill((DATATYPE*)biasPtr, (DATATYPE*)biasPtr + 2 * inputC, 0.5);
    std::fill((DATATYPE*)meanPtr, (DATATYPE*)meanPtr + inputC, 0.5);
    std::fill((DATATYPE*)varPtr, (DATATYPE*)varPtr + inputC, 0.5);
  } else {
    flags |= normalization_flags::use_global_stats;
  }
#endif
  auto bnOpDesc = batch_normalization_forward::desc(
      prop, dataMemDesc, eps, flags);
  auto bnPrimDesc = batch_normalization_forward::primitive_desc(bnOpDesc, eng);
  // create primitives and connect.
  net.clear();
  assert(bnPrimDesc.dst_desc() == outputMem.get_desc());
  net.push_back({batch_normalization_forward(bnPrimDesc),
      {{DNNL_ARG_SRC, inputMem},
      {DNNL_ARG_MEAN, meanMem},
      {DNNL_ARG_VARIANCE, varMem},
      {DNNL_ARG_SCALE_SHIFT, scaleShiftMem},
      {DNNL_ARG_DST, outputMem}}});
}

float BatchNorm::get_min_epsilon(void)
{
  return BN_MIN_EPSILON;
}

void BatchNorm::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  CHECK_NE(nullptr, scaleShiftPtr = malloc(outputs[0].dim[1] * 2));
  // create primitives.
  memory inputMem, outputMem, meanMem, varMem, scaleShiftMem;
  create_net(this, net, model->eng, model->strm,
      inputMem, outputMem, meanMem, varMem, scaleShiftMem,
      inputs[0].data_ptr, outputs[0].data_ptr,
      inputs[3].data_ptr,
      inputs[4].data_ptr,
      scaleShiftPtr,
      model->isTraining);
}

void BatchNorm::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  free(scaleShiftPtr);
  outputs[0].data_ptr = nullptr;
  scaleShiftPtr = nullptr;
}

void BatchNorm::forward(bool block)
{
  std::copy((DATATYPE*)inputs[1].data_ptr, (DATATYPE*)inputs[1].data_ptr + outputs[0].dim[1], (DATATYPE*)scaleShiftPtr);
  std::copy((DATATYPE*)inputs[2].data_ptr, (DATATYPE*)inputs[2].data_ptr + outputs[0].dim[1], (DATATYPE*)scaleShiftPtr + outputs[0].dim[1]);
  for (auto& p : net) p.first.execute(model->strm, p.second);
  if (block) model->strm.wait();
}

void Model::measure_batchnorm_cost(BatchNorm* bn)
{
  memory inputMem, outputMem, meanMem, varMem, scaleShiftMem;
  create_net(bn, net, eng, strm,
      inputMem, outputMem, meanMem, varMem, scaleShiftMem,
      inputPtr, outputPtr, runningMean, runningVar, biasPtr,
      isTraining);

  assert(inputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(outputMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(meanMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(varMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(scaleShiftMem.get_desc().get_size() <= MAX_TENSOR_SIZE);

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

  bn->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[BatchNorm]: i(%d %d %d %d) cost(%.4lf)\n",
           bn->inputs[0].dim[0], bn->inputs[0].dim[1], bn->inputs[0].dim[2],
           bn->inputs[0].dim[3], bn->runtime);
}


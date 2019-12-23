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
#include <vector>
using namespace taso;
using namespace dnnl;

static void create_net(Concat* concat, DNNLNet& net, engine& eng, stream& strm,
    std::vector<memory>& inputMems, memory& outputMem,
    std::vector<void*> inputPtrs, void* outputPtr) {
  // data descriptors.
  auto outputMemDesc = get_memory_desc(concat->outputs[0]);
  std::vector<memory::desc> inputMemDescs;
  for (int i = 0; i < concat->numInputs; i++) {
    inputMemDescs.push_back(get_memory_desc(concat->inputs[i]));
  }
  // data memories.
  outputMem = memory(outputMemDesc, eng, outputPtr);
  assert(inputMems.size() == (size_t)concat->numInputs);
  assert(inputPtrs.size() == (size_t)concat->numInputs);
  for (int i = 0; i < concat->numInputs; i++) {
    inputMems[i] = memory(inputMemDescs[i], eng, inputPtrs[i]);
  }
  // operator primitives.
  auto concatPrimDesc = concat::primitive_desc(concat->axis, inputMemDescs, eng);
  assert(concatPrimDesc.dst_desc() == outputMemDesc);
  // create primitives and connect.
  net.clear();
  DNNLNet::value_type::second_type args;
  for (int i = 0; i < concat->numInputs; i++) {
    args[DNNL_ARG_MULTIPLE_SRC + i] = inputMems[i];
  }
  args[DNNL_ARG_DST] = outputMem;
  net.push_back({::dnnl::concat(concatPrimDesc), args});
}

void Concat::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  // create primitives.
  memory outputMem;
  std::vector<memory> inputMems(numInputs);
  std::vector<void*> inputPtrs;
  for (int i = 0; i < numInputs; i++) inputPtrs.push_back(inputs[i].data_ptr);
  create_net(this, net, model->eng, model->strm,
      inputMems, outputMem,
      inputPtrs, outputs[0].data_ptr);
}

void Concat::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void Concat::forward(bool block)
{
  for (auto& p : net) p.first.execute(model->strm, p.second);
  if (block) model->strm.wait();
}

void Model::measure_concat_cost(Concat* concat)
{
  memory outputMem;
  std::vector<memory> inputMems(concat->numInputs);
  std::vector<void*> inputPtrs;
  for (int i = 0; i < concat->numInputs; i++) inputPtrs.push_back(inputPtr);
  create_net(concat, net, eng, strm,
      inputMems, outputMem,
      inputPtrs, outputPtr);

  for (const auto& m : inputMems)
    assert(m.get_desc().get_size() <= MAX_TENSOR_SIZE);
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

  concat->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Concat]: cost(%.4lf)\n", concat->runtime);
}


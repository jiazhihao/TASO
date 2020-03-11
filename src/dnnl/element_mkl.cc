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

void elementwise_kernel(int volume, OpType type,
    const Tensor& tx, const Tensor& ty, const Tensor& tz,
    const DATATYPE* x, const DATATYPE* y, DATATYPE* z) {
  int numDim = tz.numDim;
  assert(tx.numDim <= numDim);
  assert(ty.numDim <= numDim);
  assert(numDim <= 6);
  int pos[6];
#pragma omp parallel for
  for (int zid = 0; zid < volume; zid++) {
    for (int d = 0; d < numDim; d++) {
      pos[d] = (zid / tz.stride[d]) % tz.dim[d];
    }
    int xid = 0;
    int diff = numDim - tx.numDim;
    for (int d = 0; d < tx.numDim; d++) {
      xid += tx.stride[d] * pos[d + diff];
    }
    int yid = 0;
    diff = numDim - ty.numDim;
    for (int d = 0; d < ty.numDim; d++) {
      yid += ty.stride[d] * pos[d + diff];
    }

    switch (type) {
      case OP_EW_ADD:
        z[zid] = x[xid] + y[yid];
        break;
      case OP_EW_MUL:
        z[zid] = x[xid] * y[yid];
        break;
      case OP_EW_SUB:
        z[zid] = x[xid] - y[yid];
        break;
      case OP_EW_DIV:
        z[zid] = x[xid] / y[yid];
        break;
      case OP_EW_EQUAL:
        z[zid] = (x[xid] == y[yid]);
        break;
      case OP_EW_GREATER:
        z[zid] = (x[xid] > y[yid]);
        break;
      case OP_EW_LESS:
        z[zid] = (x[xid] < y[yid]);
        break;
      case OP_EW_MAX:
        z[zid] = (x[xid] > y[yid] ? x[xid] : y[yid]);
        break;
      case OP_EW_MIN:
        z[zid] = (x[xid] < y[yid] ? x[xid] : y[yid]);
        break;
      case OP_PRELU:
        z[zid] = (x[xid] >= 0 ? x[xid] : y[yid] * x[xid]);
        break;
      default:
        assert(false);
    }
  }
}

bool Element::use_kernel(void) const
{
  switch (type) {
    case OP_EW_ADD:
    case OP_EW_MUL:
      break;
    default:
      return false;
  }

  // dnnl::binary requires the same layout between output and inputs[0].
  if (!outputs[0].has_same_shape_stride_split(inputs[0])) return false;

  return true;
}

static void create_net(Element* ele, DNNLNet& net, engine& eng, stream& strm,
    memory& in0Mem, memory& in1Mem, memory& outputMem,
    void* in0Ptr, void* in1Ptr, void* outputPtr
    ) {
  // dimensions.
  assert(ele->inputs[0].volume() == ele->outputs[0].volume());
  assert(ele->inputs[1].volume() == ele->outputs[0].volume());
  int numDim = ele->outputs[0].numDim;
  if (ele->use_kernel()) {
    // data descriptors.
    auto in0MemDesc = get_memory_desc(ele->inputs[0], numDim);
    auto in1MemDesc = get_memory_desc(ele->inputs[1], numDim);
    auto outputMemDesc = get_memory_desc(ele->outputs[0], numDim);
    // data memories.
    in0Mem = memory(in0MemDesc, eng, in0Ptr);
    in1Mem = memory(in1MemDesc, eng, in1Ptr);
    outputMem = memory(outputMemDesc, eng, outputPtr);
    // operator primitives.
    algorithm ops_algo = algorithm::binary_add;
    if (ele->type == OP_EW_MUL) ops_algo = algorithm::binary_mul;
    else assert(ele->type == OP_EW_ADD);
    auto eleOpDesc = binary::desc(ops_algo, in0MemDesc, in1MemDesc, outputMemDesc);
    auto elePrimDesc = binary::primitive_desc(eleOpDesc, eng);
    // create primitives and connect.
    net.clear();
    net.push_back({binary(elePrimDesc),
        {{DNNL_ARG_SRC_0, in0Mem},
        {DNNL_ARG_SRC_1, in1Mem},
        {DNNL_ARG_DST, outputMem}}});
  } else {
    // No preprocessing for our customized kernel
  }
}

void Element::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  // create primitives.
  memory in0Mem, in1Mem, outputMem;
  create_net(this, net, model->eng, model->strm,
      in0Mem, in1Mem, outputMem,
      inputs[0].data_ptr, inputs[1].data_ptr, outputs[0].data_ptr);
}

void Element::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void Element::forward(bool block)
{
  if (use_kernel()) {
    for (auto& p : net) p.first.execute(model->strm, p.second);
    if (block) model->strm.wait();
  } else {
    elementwise_kernel(outputs[0].volume(), type,
        inputs[0], inputs[1], outputs[0],
        (DATATYPE*)inputs[0].data_ptr,
        (DATATYPE*)inputs[1].data_ptr,
        (DATATYPE*)outputs[0].data_ptr);
  }
}

void Model::measure_element_cost(Element* ele)
{
  memory in0Mem, in1Mem, outputMem;
  create_net(ele, net, eng, strm,
      in0Mem, in1Mem, outputMem,
      inputPtr, biasPtr, outputPtr);

  // measure.
  uint64_t beg = 0;
  if (ele->use_kernel()) {
    assert(in0Mem.get_desc().get_size() <= MAX_TENSOR_SIZE);
    assert(in1Mem.get_desc().get_size() <= MAX_TENSOR_SIZE);
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
      elementwise_kernel(ele->outputs[0].volume(), ele->type,
          ele->inputs[0], ele->inputs[1], ele->outputs[0],
          (DATATYPE*)inputPtr,
          (DATATYPE*)biasPtr,
          (DATATYPE*)outputPtr);
    }
  }
  auto end = microsecond_timer();

  ele->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Element]: i(%d %d %d %d) type(%d) cost(%.4lf)\n",
           ele->inputs[0].dim[0], ele->inputs[0].dim[1], ele->inputs[0].dim[2],
           ele->inputs[0].dim[3], ele->type, ele->runtime);
}


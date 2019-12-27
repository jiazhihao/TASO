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

static void create_net(Matmul* mm, DNNLNet& net, engine& eng, stream& strm,
#ifdef DNNL_NO_MATMUL
    Matmul::BLASGEMMParams& params,
#endif
    memory& aMem, memory& bMem, memory& cMem, void* aPtr, void* bPtr, void* cPtr) {
  // dimensions.
  int numDim = mm->outputs[0].numDim;
  int m = mm->inputs[0].dim[numDim-2];
  int n = mm->inputs[1].dim[numDim-1];
  int k = mm->inputs[0].dim[numDim-1];
  int b = 1;
  if (numDim > 2) {
    for (int i = 0; i < numDim - 2; i++) {
      b *= mm->outputs[0].dim[i];
      assert(mm->inputs[0].dim[i] == mm->outputs[0].dim[i]);
      assert(mm->inputs[1].dim[i] == mm->outputs[0].dim[i]);
    }
  }
  // data sizes.
  const memory::dims aSize = { b, m, k };
  const memory::dims bSize = { b, k, n };
  const memory::dims cSize = { b, m, n };
  const memory::dims aStride = { m * k, mm->inputs[0].stride[numDim-2], mm->inputs[0].stride[numDim-1] };
  const memory::dims bStride = { k * n, mm->inputs[1].stride[numDim-2], mm->inputs[1].stride[numDim-1] };
  const memory::dims cStride = { m * n, mm->outputs[0].stride[numDim-2], mm->outputs[0].stride[numDim-1] };
  // data descriptors.
  auto aMemDesc = memory::desc(aSize, DNNL_DEF_DTYPE, aStride);
  auto bMemDesc = memory::desc(bSize, DNNL_DEF_DTYPE, bStride);
  auto cMemDesc = memory::desc(cSize, DNNL_DEF_DTYPE, cStride);
  // data memories.
  aMem = memory(aMemDesc, eng, aPtr);
  bMem = memory(bMemDesc, eng, bPtr);
  cMem = memory(cMemDesc, eng, cPtr);
#ifndef DNNL_NO_MATMUL
  // operator primitives.
  auto mmOpDesc = matmul::desc(aMemDesc, bMemDesc, cMemDesc);
  auto mmPrimDesc = matmul::primitive_desc(mmOpDesc, eng);
  if (mm->activation != AC_MODE_NONE) {
    auto mmAttr = get_activation_attr(mm->activation);
    mmPrimDesc = matmul::primitive_desc(mmOpDesc, mmAttr, eng);
  }
  // create primitives and connect.
  net.clear();
  net.push_back({matmul(mmPrimDesc),
      {{DNNL_ARG_SRC, aMem}, {DNNL_ARG_WEIGHTS, bMem}, {DNNL_ARG_DST, cMem}}});
#else  // DNNL_NO_MATMUL
  // BLAS parameters.
  params.batch = b;
  params.m = m;
  params.n = n;
  params.k = k;
  if (mm->inputs[0].stride[numDim-2] == 1) {
    params.transA = 't';
    params.lda = mm->inputs[0].stride[numDim-1];
  } else {
    assert(mm->inputs[0].stride[numDim-1] == 1);
    params.transA = 'n';
    params.lda = mm->inputs[0].stride[numDim-2];
  }
  if (mm->inputs[1].stride[numDim-2] == 1) {
    params.transB = 't';
    params.ldb = mm->inputs[1].stride[numDim-1];
  } else {
    assert(mm->inputs[1].stride[numDim-1] == 1);
    params.transB = 'n';
    params.ldb = mm->inputs[1].stride[numDim-2];
  }
  assert(mm->outputs[0].stride[numDim-1] == 1);
  params.ldc = mm->outputs[0].stride[numDim-2];
  // activation primitive.
  net.clear();
  if (mm->activation != AC_MODE_NONE) {
    algorithm ops_algo = algorithm::eltwise_relu;  // relu as default
    float ops_alpha = 0.0f;  // relu negative slope
    float ops_beta = 0.0f;
    switch (mm->activation) {
      case AC_MODE_SIGMOID:
        ops_algo = algorithm::eltwise_logistic;
        // alpha and beta ignored.
        break;
      case AC_MODE_TANH:
        ops_algo = algorithm::eltwise_tanh;
        // alpha and beta ignored.
        break;
      case AC_MODE_RELU:
      default:
        break;
    }
    auto actOpDesc = eltwise_forward::desc(
        prop_kind::forward_inference, ops_algo, cMemDesc, ops_alpha, ops_beta);
    auto actPrimDesc = eltwise_forward::primitive_desc(actOpDesc, eng);
    net.push_back({eltwise_forward(actPrimDesc),
      {{DNNL_ARG_SRC, cMem}, {DNNL_ARG_DST, cMem}}});
  }
#endif  // DNNL_NO_MATMUL
}

void Matmul::map(void)
{
  // allocate tensors
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  CHECK_NE(nullptr, outputs[0].data_ptr = malloc(outputSize));
  // create primitives.
  memory aMem, bMem, cMem;
  create_net(this, net, model->eng, model->strm,
#ifdef DNNL_NO_MATMUL
      params,
#endif
      aMem, bMem, cMem, inputs[0].data_ptr, inputs[1].data_ptr, outputs[0].data_ptr);
}

void Matmul::unmap(void)
{
  // clear primitives
  net.clear();
  // free tensors
  free(outputs[0].data_ptr);
  outputs[0].data_ptr = nullptr;
}

void Matmul::forward(bool block)
{
#ifdef DNNL_NO_MATMUL
  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int b = 0; b < params.batch; b++) {
    CHECK_EQ(dnnl_success,
        dnnl_sgemm(params.transA, params.transB,
          params.m, params.n, params.k, alpha,
          (DATATYPE*)inputs[0].data_ptr + b * params.m * params.k, params.lda,
          (DATATYPE*)inputs[1].data_ptr + b * params.k * params.n, params.ldb, beta,
          (DATATYPE*)outputs[0].data_ptr + b * params.m * params.n, params.ldc));
  }
#endif
  for (auto& p : net) p.first.execute(model->strm, p.second);
  if (block) model->strm.wait();
}

void Matmul::set_layout(void)
{
  // MKL uses row-major.
  int numDim = outputs[0].numDim;
  outputs[0].stride[numDim-1] = 1;
  outputs[0].stride[numDim-2] = outputs[0].dim[numDim-1];
  int size = outputs[0].dim[numDim-2] * outputs[0].dim[numDim-1];
  for (int i = numDim-3; i >= 0; i--) {
    outputs[0].stride[i] = size;
    size *= outputs[0].dim[i];
  }
  assert(size == outputs[0].volume());
}

void Model::measure_matmul_cost(Matmul* mm)
{
  memory aMem, bMem, cMem;
#ifdef DNNL_NO_MATMUL
  Matmul::BLASGEMMParams params;
#endif
  create_net(mm, net, eng, strm,
#ifdef DNNL_NO_MATMUL
      params,
#endif
      aMem, bMem, cMem, inputPtr, filterPtr, outputPtr);

  assert(aMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(bMem.get_desc().get_size() <= MAX_TENSOR_SIZE);
  assert(cMem.get_desc().get_size() <= MAX_TENSOR_SIZE);

  // measure.
  uint64_t beg = 0;
  for (int i = 0; i < WARMUP_TIMES + REPEAT_TIMES; i++) {
    if (i == WARMUP_TIMES) {
      beg = microsecond_timer();
    }
#ifdef DNNL_NO_MATMUL
    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int b = 0; b < params.batch; b++) {
      CHECK_EQ(dnnl_success,
          dnnl_sgemm(params.transA, params.transB,
            params.m, params.n, params.k, alpha,
            (DATATYPE*)inputPtr + b * params.m * params.k, params.lda,
            (DATATYPE*)filterPtr + b * params.k * params.n, params.ldb, beta,
            (DATATYPE*)outputPtr + b * params.m * params.n, params.ldc));
    }
#endif
    for (auto& p : net) {
      p.first.execute(strm, p.second);
    }
    strm.wait();
  }
  auto end = microsecond_timer();

  mm->runtime = (end - beg) / 1.e3 / REPEAT_TIMES;  // milliseconds
  if (print_cost)
    printf("  measure[Matmul]: %s %s acti(%d) cost(%.4lf)\n",
           mm->inputs[0].to_string("input").c_str(),
           mm->inputs[1].to_string("weight").c_str(),
           mm->activation, mm->runtime);
}


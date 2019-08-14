/* Copyright 2018 Stanford
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

#include "xflow/ops.h"
#include "xflow/cuda_helper.h"
using namespace XFlow;

void Concat::map(void)
{
  size_t outputSize = sizeof(DATATYPE) * outputs[0].volume();
  checkCUDA(cudaMalloc(&outputs[0].data_ptr, outputSize));
}

void Concat::unmap(void)
{
  checkCUDA(cudaFree(outputs[0].data_ptr));
}

__global__
void assign_with_stride(DATATYPE* dst,
                        const DATATYPE* src,
                        int num_blocks,
                        int dst_blk_size,
                        int src_blk_size)
{
  assert(src_blk_size <= dst_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * src_blk_size)
  {
    int blk_idx = i / src_blk_size;
    int blk_offset = i % src_blk_size;
    int src_offset = blk_idx * src_blk_size + blk_offset;
    int dst_offset = blk_idx * dst_blk_size + blk_offset;
    dst[dst_offset] = src[src_offset];
  }
}

void Concat::forward(bool block)
{
  int offset = 0;
  for (int i = 0; i < numInputs; i++)
    if (needCopy[i]) {
    int dst_blk_size = 1, src_blk_size = 1, num_blocks = 1;
    for (int j = inputs[i].numDim-1; j >= 0; j--)
      if (j >= axis) {
        dst_blk_size *= outputs[0].dim[j];
        src_blk_size *= inputs[i].dim[j];
      } else {
        num_blocks *= outputs[0].dim[j];
      }
    assert(inputs[i].data_ptr != NULL);
    assign_with_stride<<<GET_BLOCKS(num_blocks*src_blk_size), CUDA_NUM_THREADS>>>(
        ((DATATYPE*)outputs[0].data_ptr) + offset, (DATATYPE*)inputs[i].data_ptr,
        num_blocks, dst_blk_size, src_blk_size);
    offset += src_blk_size;
  }
  if (block)
    checkCUDA(cudaDeviceSynchronize());
  //FIXME
  //DATATYPE* print_vals = (DATATYPE*) malloc(outputs[0].volume() * sizeof(DATATYPE));
  //checkCUDA(cudaMemcpy(print_vals, outputs[0].data_ptr, outputs[0].volume() * sizeof(DATATYPE), cudaMemcpyDefault));
  //for (int i = 0; i < outputs[0].volume(); i++)
  //  printf("output[%d]: %.4lf\n", i, print_vals[i]);
  //for (int i = 0; i < numInputs; i++) {
  //  checkCUDA(cudaMemcpy(print_vals, inputs[i].data_ptr, inputs[i].volume() * sizeof(DATATYPE), cudaMemcpyDefault));
  //  printf("concat_forward: inputs[%d].ptr=%p\n", i, inputs[i].data_ptr);
  //  for (int j = 0; j < inputs[i].volume(); j++)
  //    printf("input[%d][%d]: %.4lf\n", i, j, print_vals[j]);
  //}
}

void Model::measure_concat_cost(Concat* concat)
{
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int i = 0; i < REPEAT_TIMES; i++) {
    int offset = 0;
    // TODO: remove needCopy and should not include operators
    // that can be preproceed
    for (int j = 0; j < concat->numInputs; j++) 
      if (concat->needCopy[j]) {
        int dst_blk_size = 1, src_blk_size = 1, num_blocks = 1;
        for (int d = concat->inputs[j].numDim-1; d >= 0; d--)
          if (d >= concat->axis) {
            dst_blk_size *= concat->outputs[0].dim[d];
            src_blk_size *= concat->inputs[j].dim[d];
          } else {
            num_blocks *= concat->outputs[0].dim[d];
          }
        assign_with_stride<<<GET_BLOCKS(num_blocks*src_blk_size), CUDA_NUM_THREADS>>>(
            ((DATATYPE*)outputPtr) + offset, (DATATYPE*)inputPtr,
            num_blocks, dst_blk_size, src_blk_size);
        offset += src_blk_size;
      }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  concat->runtime = milliseconds / REPEAT_TIMES;
  if (print_cost)
    printf("  measure[Concat]: cost(%.4lf)\n", concat->runtime);
}


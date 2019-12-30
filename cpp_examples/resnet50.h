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

#ifndef _CPP_EXAMPLES_RESNET50_H_
#define _CPP_EXAMPLES_RESNET50_H_

TensorHandle resnet_block(Graph* graph, const TensorHandle input, int strideH, int strideW, int outChannels) {
  TensorHandle t = input;
  auto w1 = new_random_weight(graph, { outChannels, t->dim[1], 1, 1 });
  t = graph->conv2d(t, w1, 1, 1, PD_MODE_SAME, AC_MODE_RELU);
  auto w2 = new_random_weight(graph, { outChannels, t->dim[1], 3, 3 });
  t = graph->conv2d(t, w2, strideH, strideW, PD_MODE_SAME, AC_MODE_RELU);
  auto w3 = new_random_weight(graph, { 4 * outChannels, t->dim[1], 1, 1 });
  t = graph->conv2d(t, w3, 1, 1, PD_MODE_SAME);
  auto inp = input;
  if (strideH > 1 || inp->dim[1] != 4 * outChannels) {
    auto w4 = new_random_weight(graph, { 4 * outChannels, inp->dim[1], 1, 1 });
    inp = graph->conv2d(inp, w4, strideH, strideW, PD_MODE_SAME, AC_MODE_RELU);
  }
  return graph->relu(graph->element(OP_EW_ADD, inp, t));
}

Graph* resnet50(float alpha, int budget, bool printSubst = false) {
  Graph *graph = new Graph();
  auto inp = new_input(graph, { 1, 64, 56, 56 });
  auto t = inp;
  int stride = 1;
  for (int i = 0; i < 3; i++) {
    t = resnet_block(graph, t, stride, stride, 64);
  }
  stride = 2;
  for (int i = 0; i < 4; i++) {
    t = resnet_block(graph, t, stride, stride, 128);
    stride = 1;
  }
  stride = 2;
  for (int i = 0; i < 6; i++) {
    t = resnet_block(graph, t, stride, stride, 256);
    stride = 1;
  }
  stride = 2;
  for (int i = 0; i < 3; i++) {
    t = resnet_block(graph, t, stride, stride, 512);
    stride = 1;
  }
  return graph->optimize(alpha, budget, printSubst);
}

#endif

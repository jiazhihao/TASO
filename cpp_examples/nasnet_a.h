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

#ifndef _CPP_EXAMPLES_NASNET_A_H_
#define _CPP_EXAMPLES_NASNET_A_H_

#include <vector>

TensorHandle squeeze(Graph* graph, const TensorHandle input, int outChannels) {
  auto weight = new_random_weight(graph, { outChannels, input->dim[1], 1, 1 });
  return graph->conv2d(input, weight, 1, 1, PD_MODE_SAME, AC_MODE_RELU);
}

TensorHandle fit(Graph* graph, const TensorHandle current, const TensorHandle input) {
  if (input->dim[2] == current->dim[2]) {
    return squeeze(graph, input, current->dim[1]);
  }
  auto weight = new_random_weight(graph, { current->dim[1], input->dim[1], 3, 3 });
  return graph->conv2d(input, weight, 2, 2, PD_MODE_SAME, AC_MODE_RELU);
}

TensorHandle separable_conv(Graph* graph, const TensorHandle input, int outChannels,
    int kernelH, int kernelW, int strideH, int strideW,
    PaddingMode padding, ActiMode activation = AC_MODE_NONE) {
  assert(input->dim[1] % outChannels == 0);
  auto w1 = new_random_weight(graph, { outChannels, input->dim[1] / outChannels, kernelH, kernelW });
  auto t = graph->conv2d(input, w1, strideH, strideW, padding);
  auto w2 = new_random_weight(graph, { outChannels, t->dim[1], 1, 1 });
  return graph->conv2d(t, w2, 1, 1, PD_MODE_SAME, activation);
}

TensorHandle normal_cell(Graph* graph, TensorHandle prev, TensorHandle cur, int outChannels) {
  cur = squeeze(graph, cur, outChannels);
  prev = fit(graph, cur, prev);
  std::vector<TensorHandle> ts;
  ts.push_back(separable_conv(graph, cur, outChannels, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(cur);
  ts.push_back(separable_conv(graph, prev, outChannels, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, cur, outChannels, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(graph->pool2d_avg(cur, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(prev);
  ts.push_back(graph->pool2d_avg(prev, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(graph->pool2d_avg(prev, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, prev, outChannels, 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, prev, outChannels, 3, 3, 1, 1, PD_MODE_SAME));
  assert(ts.size() == 10);
  std::vector<TensorHandle> outputs;
  for (int i = 0; i < 5; i++) {
    outputs.push_back(graph->element(OP_EW_ADD, ts[2 * i], ts[2 * i + 1]));
  }
  return graph->concat(1, outputs.size(), outputs.data());
}

TensorHandle reduction_cell(Graph* graph, TensorHandle prev, TensorHandle cur, int outChannels) {
  cur = squeeze(graph, cur, outChannels);
  prev = fit(graph, cur, prev);
  std::vector<TensorHandle> ts;
  std::vector<TensorHandle> outputs;
  ts.push_back(separable_conv(graph, prev, outChannels, 7, 7, 2, 2, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, cur, outChannels, 5, 5, 2, 2, PD_MODE_SAME));
  outputs.push_back(graph->element(OP_EW_ADD, ts[0], ts[1]));
  ts.push_back(graph->pool2d_max(cur, 3, 3, 2, 2, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, prev, outChannels, 7, 7, 2, 2, PD_MODE_SAME));
  outputs.push_back(graph->element(OP_EW_ADD, ts[2], ts[3]));
  ts.push_back(graph->pool2d_avg(cur, 3, 3, 2, 2, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, prev, outChannels, 5, 5, 2, 2, PD_MODE_SAME));
  outputs.push_back(graph->element(OP_EW_ADD, ts[4], ts[5]));
  ts.push_back(graph->pool2d_max(cur, 3, 3, 2, 2, PD_MODE_SAME));
  ts.push_back(separable_conv(graph, outputs[0], outChannels, 3, 3, 1, 1, PD_MODE_SAME));
  outputs.push_back(graph->element(OP_EW_ADD, ts[6], ts[7]));
  ts.push_back(graph->pool2d_avg(outputs[0], 3, 3, 1, 1, PD_MODE_SAME));
  ts.push_back(outputs[1]);
  outputs.push_back(graph->element(OP_EW_ADD, ts[8], ts[9]));
  return graph->concat(1, outputs.size(), outputs.data());
}

Graph* nasnet_a(float alpha, int budget, bool printSubst = false) {
  Graph *graph = new Graph();
  auto inp = new_input(graph, { 1, 3, 224, 224 });
  auto weight = new_random_weight(graph, { 64, 3, 7, 7 });
  inp = graph->conv2d(inp, weight, 2, 2, PD_MODE_SAME, AC_MODE_RELU);
  inp = graph->pool2d_max(inp, 3, 3, 2, 2, PD_MODE_SAME);
  int outChannels = 128;
  for (int i = 0; i < 3; i++) {
    auto prev = inp;
    auto cur = inp;
    for (int j = 0; j < 5; j++) {
      auto t = normal_cell(graph, prev, cur, outChannels);
      prev = cur;
      cur = t;
    }
    outChannels *= 2;
    inp = reduction_cell(graph, prev, cur, outChannels);
  }
  return graph->optimize(alpha, budget, printSubst);
}

#endif

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

#ifndef _CPP_EXAMPLES_NASRNN_H_
#define _CPP_EXAMPLES_NASRNN_H_

#include <vector>

#define NASRNN_HIDDEN_SIZE 512
#define NASRNN_LENGTH 5

TensorHandle combine(Graph* graph, const TensorHandle x, const TensorHandle h) {
  auto w1 = new_random_weight(graph, { NASRNN_HIDDEN_SIZE, x->dim[1] });
  auto w2 = new_random_weight(graph, { NASRNN_HIDDEN_SIZE, h->dim[1] });
  return graph->element(OP_EW_ADD, graph->matmul(x, w1), graph->matmul(h, w2));
}

TensorHandle nas_node(Graph* graph, const TensorHandle inp, const TensorHandle x) {
  std::vector<TensorHandle> ts;
  for (int i = 0; i < 8; i++)
    ts.push_back(combine(graph, x, inp));
  std::vector<TensorHandle> midts;
  midts.push_back(graph->element(OP_EW_ADD, graph->relu(ts[0]), graph->sigmoid(ts[3])));
  midts.push_back(graph->element(OP_EW_ADD, graph->sigmoid(ts[1]), graph->tanh(ts[2])));
  midts.push_back(graph->element(OP_EW_MUL, graph->sigmoid(ts[4]), graph->tanh(ts[5])));
  midts.push_back(graph->element(OP_EW_MUL, graph->sigmoid(ts[6]), graph->relu(ts[7])));
  midts.push_back(graph->element(OP_EW_ADD, graph->sigmoid(midts[1]), graph->tanh(midts[2])));
  midts.push_back(graph->element(OP_EW_MUL, graph->tanh(midts[0]), graph->tanh(midts[3])));
  midts.push_back(graph->element(OP_EW_MUL, graph->tanh(midts[4]), graph->tanh(midts[5])));
  return graph->tanh(midts[6]);
}

Graph* nasrnn(float alpha, int budget, bool printSubst = false) {
  Graph *graph = new Graph();
  std::vector<TensorHandle> xs;
  for (int i = 0; i < NASRNN_LENGTH; i++) {
    xs.push_back(new_input(graph, { 1, NASRNN_HIDDEN_SIZE }));
  }
  auto state = new_random_weight(graph, { 1, NASRNN_HIDDEN_SIZE });
  for (int i = 0; i < NASRNN_LENGTH; i++) {
    state = nas_node(graph, state, xs[i]);
  }
  return graph->optimize(alpha, budget, printSubst);
}

#undef NASRNN_HIDDEN_SIZE
#undef NASRNN_LENGTH

#endif

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

#ifndef _CPP_EXAMPLES_BERT_H_
#define _CPP_EXAMPLES_BERT_H_

TensorHandle attention(Graph* graph, const TensorHandle input, int heads) {
  int d_model = input->dim[1];
  int d_k = d_model / heads;
  assert(input->dim[1] % heads == 0);
  TensorHandle weights[3];
  for (int i = 0; i < 3; i++) {
    weights[i] = new_random_weight(graph, { d_model, d_model });
  }
  // compute query, key, value tensors
  auto q = graph->matmul(input, weights[0]);
  auto k = graph->matmul(input, weights[1]);
  auto v = graph->matmul(input, weights[2]);
  // reshape query, key, value to multiple heads
  q = graph->reshape(q, { -1, heads, d_k });
  k = graph->reshape(k, { -1, heads, d_k });
  v = graph->reshape(v, { -1, heads, d_k });
  // transpose query, key, value for batched matmul
  q = graph->transpose(q, { 1, 0, 2 }, true);
  k = graph->transpose(k, { 1, 2, 0 }, true);
  v = graph->transpose(v, { 1, 0, 2 }, true);
  // perform matrix multiplications
  auto logits = graph->matmul(q, k);
  auto output = graph->matmul(logits, v);
  // transpose the output back
  output = graph->transpose(output, { 1, 0, 2 }, true);
  output = graph->reshape(output, { input->dim[0], input->dim[1] });

  // a final linear layer
  auto linear = new_random_weight(graph, { d_model, d_model });
  output = graph->matmul(output, linear);
  return output;
}

Graph* bert(float alpha, int budget, bool printSubst = false) {
  const int seq_length = 64;
  const int hidden_dims = 1024;
  Graph *graph = new Graph();
  auto inp = new_input(graph, { seq_length, hidden_dims });
  inp = graph->relu(inp);
  auto t = inp;
  for (int i = 0; i < 8; i++) {
    t = attention(graph, t, 16);
  }
  return graph->optimize(alpha, budget, printSubst);
}

#endif

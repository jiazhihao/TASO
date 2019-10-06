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
#include "taso/substitution.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace taso;

const Op Op::INVALID_OP = Op();
const SplitInfo SplitInfo::NO_SPLIT = SplitInfo();

/*
bool Op::operator==(const Op& b)
{
  if (guid != b.guid) return false;
  return (ptr == b.ptr);
}

bool Op::operator<(const Op& b)
{
  if (guid != b.guid) return guid < b.guid;
  return ptr < b.ptr;
}
*/

Op::Op(void)
{
  guid = GUID_INVALID;
  ptr = NULL;
}

Edge::Edge(void)
: srcOp(Op::INVALID_OP), dstOp(Op::INVALID_OP), srcIdx(-1), dstIdx(-1)
{}

Edge::Edge(Op _srcOp, Op _dstOp, int _srcIdx, int _dstIdx)
: srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx)
{}

SrcEdge::SrcEdge(int _idx, Op _op)
: idx(_idx), op(_op)
{}

/*
bool Tensor::operator==(const Tensor& b)
{
  if (numDim != b.numDim) return false;
  for (int i = 0; i < numDim; i++)
    if (dim[i] != b.dim[i]) return false;
  if (idx != b.idx) return false;
  if (op.guid != b.op.guid) return false;
  return true;
}
*/

OpBase::OpBase(Model* _model, OpType _type)
: numInputs(0), model(_model), type(_type), runtime(0.0f)
{
  // Assume only constant operator can take no inputs
  assert(type == OP_CONSTANT_POOL);
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(Tensor _input, Model* _model, OpType _type)
: numInputs(1), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(Tensor _input0, Tensor _input1, Model* _model, OpType _type)
: numInputs(2), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(Tensor _input0, Tensor _input1, Tensor _input2, Tensor _input3,
               Tensor _input4, Model* _model, OpType _type)
: numInputs(5), model(_model), type(_type), runtime(0.0f)
{
  inputs[0] = _input0;
  inputs[1] = _input1;
  inputs[2] = _input2;
  inputs[3] = _input3;
  inputs[4] = _input4;
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

OpBase::OpBase(int n, Tensor* _inputs, Model* _model, OpType _type)
: numInputs(n), model(_model), type(_type), runtime(0.0f)
{
  assert(n <= MAX_NUM_INPUTS);
  for (int i = 0; i < n; i++)
    inputs[i] = _inputs[i];
  for (int i = 0; i < MAX_NUM_OUTPUTS; i++) {
    outputs[i].numDim = 0;
    for (int j = 0; j < MAX_DIM; j++)
      outputs[i].dim[j] = 0;
  }
}

bool OpBase::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    case PM_OP_TYPE:
      *value = (int) type;
      return true;
    case PM_NUM_INPUTS:
      *value = numInputs;
      return true;
    case PM_NUM_OUTPUTS:
      *value = numOutputs;
      return true;
    default:
      return false;
  }
}

bool OpBase::get_input_parameter(TNParameter tnp, DIMParameter dim, int* value)
{
  int inputIdx = 0, dimIdx = 0;
  switch (tnp) {
    case IN_5:
      inputIdx++;
    case IN_4:
      inputIdx++;
    case IN_3:
      inputIdx++;
    case IN_2:
      inputIdx++;
    case IN_1:
      inputIdx++;
    case IN_0:
      break;
    default:
      return false;
  }
  if (inputIdx >= numInputs) return false;
  switch (dim) {
    case DIM_3:
      dimIdx ++;
    case DIM_2:
      dimIdx ++;
    case DIM_1:
      dimIdx ++;
    case DIM_0:
      break;
    case DIM_ND:
      *value = inputs[inputIdx].numDim;
      return true;
    default:
      return false;
  }
  if (dimIdx >= inputs[inputIdx].numDim) return false;
  *value = inputs[inputIdx].dim[dimIdx];
  return true;
}

static Model* model_singleton = NULL;

Graph::Graph()
: totalCost(-1.0f)
{
  if (model_singleton == NULL) {
    model_singleton = new Model();
  }
  model = model_singleton;
  model->print_cost = false;
  //size_t inputSize = sizeof(DATATYPE) * n * c * h * w;
  //checkCUDA(cudaMalloc(&input.ptr, inputSize));
  //printf("Initialize a graph\n");
}

void Graph::print_measurements(void)
{
  model->print_cost = true;
}

TensorHandle Graph::new_input(int ndim, const int* dims)
{
  TensorHandle t = new Tensor(ndim, dims, GUID_INPUT);
  t = input_wrapper(t);
  return t;
}

TensorHandle Graph::new_weight(int ndim, const int* dims, const DATATYPE* weight_initial)
{
  DATATYPE* weight_ptr = NULL;
  if (weight_initial != NULL) {
    int total_size = sizeof(DATATYPE);
    for (int i = 0; i < ndim; i++)
      total_size *= dims[i];
    weight_ptr = (DATATYPE*) model->allocate_memory(total_size, weight_initial);
  }
  TensorHandle t = new Tensor(ndim, dims, GUID_WEIGHT, weight_ptr);
  t = weight_wrapper(t);
  return t;
}

TensorHandle Graph::new_weight(const Tensor& weight)
{
  TensorHandle t = new Tensor(weight);
  t->op.guid = GUID_WEIGHT;
  t->op.ptr = NULL;
  t->idx = 0;
  t->data_ptr = (DATATYPE*) model->allocate_memory(
      weight.volume() * sizeof(DATATYPE), (DATATYPE*) weight.data_ptr);
  t = weight_wrapper(t);
  return t;
}

Graph* Graph::optimize(float alpha, int budget)
{
  std::vector<GraphXfer*> xfers;
  xfers.push_back(GraphXfer::create_conv_relu(model, 1, 1, PD_MODE_SAME));
  xfers.push_back(GraphXfer::create_conv_relu(model, 1, 1, PD_MODE_VALID));
  xfers.push_back(GraphXfer::create_conv_relu(model, 2, 2, PD_MODE_SAME));
  xfers.push_back(GraphXfer::create_conv_relu(model, 2, 2, PD_MODE_VALID));
  xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_enlarge_merge_convs(model, AC_MODE_RELU));
  xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_merge_group_convs(model, 1, 1, AC_MODE_RELU));
  xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_NONE));
  xfers.push_back(GraphXfer::create_merge_group_convs(model, 2, 2, AC_MODE_RELU));

  //xfers.push_back(create_avg_pool_conv(model));
  //xfers.push_back(create_two_pools(model));
  //xfers.push_back(create_merge_seperable_convs(model));
  GraphXfer::load_graph_xfer_from_pb_file(model, xfers, "graph_subst.pb");
  //xfers.push_back(create_fuse_conv_batch_xfer(model));
  //xfers.push_back(create_fuse_conv_relu_xfer(model));
  //xfers.push_back(create_merge_conv_xfer(model));
  //xfers.push_back(create_exclusive_concat_xfer(model));
  //xfers.push_back(create_enlarge_conv_xfer(model));
  //xfers.push_back(create_resnet_merge_xfer(model));

  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  std::set<size_t> hashmap;
  candidates.push(this);
  hashmap.insert(hash());
  Graph *bestGraph = this;
  float bestCost = total_cost();
  //printf("MetaFlow Cost = %.4lfms\n", bestCost);
  printf("Input graph: end-to-end execution time =\n"
         "%.8lf ms (average of 100 runs)\n", run());
  print_costs();

  int counter = 0;
  int maxNumOps = inEdges.size();
  //long long start_time = microsecond_timer();
  ofstream timer_fs;
  timer_fs.open("timer.txt");
  printf("\n        ===== Start Cost-Based Backtracking Search =====\n");
  while (!candidates.empty()) {
    Graph *subGraph = candidates.top();
    candidates.pop();
    if (subGraph->total_cost() < bestCost) {
      delete bestGraph;
      bestCost = subGraph->total_cost();
      bestGraph = subGraph;
    }
    if (counter > budget) {
      // TODO: free all remaining candidates when budget exhausted 
      break;
    }
    if (counter % 1 == 0) {
      printf("        [%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
      //timer_fs << microsecond_timer() - start_time << ", " << bestCost << std::endl;
    }
    counter ++;
    for (size_t i = 0; i < xfers.size(); i++) {
      //for (size_t j = 0; j < xfers[i]->srcOps.size(); j++) {
      //  printf("srcOps[%zu]: type(%d)\n", j, xfers[i]->srcOps[j]->type);
      //}
      //for (size_t j = 0; j < xfers[i]->dstOps.size(); j++) {
      //  printf("dstOps[%zu]: type(%d)\n", j, xfers[i]->dstOps[j]->type);
      //}
      xfers[i]->run(0, subGraph, candidates, hashmap, bestCost * alpha, 2 * maxNumOps);
    }
    if (bestGraph != subGraph) {
      delete subGraph;
    }
  }
  bestGraph = bestGraph->preprocess_weights();
  printf("        ===== Finish Cost-Based Backtracking Search =====\n\n");
  //printf("bestCost = %.4lf\n", bestGraph->total_cost());
  printf("Optimized graph: end-to-end execution time =\n");
  printf("%.8lf ms (average of 100 runs)\n", bestGraph->run());
  bestGraph->print_costs();

  return bestGraph;
}

Graph* Graph::preprocess_weights(void)
{
  Graph* newGraph = new Graph();
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  // Step 1: clone the input graph
  for (opIt = inEdges.begin(); opIt != inEdges.end(); opIt++)
  {
    const std::set<Edge, EdgeCompare>& list = opIt->second;
    std::set<Edge, EdgeCompare>::const_iterator it;
    for (it = list.begin(); it != list.end(); it++)
      newGraph->add_edge(it->srcOp, it->dstOp, it->srcIdx, it->dstIdx);
  }
  // Step 2: iteratively process the weights
  while (true) {
    bool change = false;
    for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); opIt++) {
      if (opIt->first.ptr->type == OP_INPUT || opIt->first.ptr->type == OP_WEIGHT)
        continue;
      bool allWeights = true;
      const std::set<Edge, EdgeCompare>& list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (it->srcOp.ptr->type != OP_WEIGHT) {
          allWeights = false;
          break;
        }
      if (allWeights) {
        // Preprocess weights
        // Currently assume the op has single output
        Op op = opIt->first;
        assert(op.ptr->numOutputs == 1);
        // map and execute the operator to get the output weights
        for (it = list.begin(); it != list.end(); it++) {
          assert(it->srcOp.ptr->outputs[it->srcIdx].data_ptr != NULL);
          assert(op.ptr->inputs[it->dstIdx].has_same_shape_stride_split(
              it->srcOp.ptr->outputs[it->srcIdx]));
          op.ptr->inputs[it->dstIdx].data_ptr =
              it->srcOp.ptr->outputs[it->srcIdx].data_ptr;
        }
        op.ptr->map();
        op.ptr->forward(true/*block*/);
        TensorHandle tensor = newGraph->new_weight(op.ptr->outputs[0]);
        newGraph->replace_node(op, tensor->op);
        op.ptr->unmap();
        newGraph->remove_node(op);
        change = true;
        break;
      }
    }
    // Stop if we didn't make any change
    if (!change)
      break;
  }
  // Remove isolated nodes
  std::map<Op, int, OpCompare> todos;
  std::vector<Op> weightList;
  std::set<Op, OpCompare> weightOps;
  for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); opIt++) {
    int cnt = 0;
    const std::set<Edge, EdgeCompare>& inList = opIt->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid != GUID_WEIGHT) cnt ++;
    }
    todos[opIt->first] = cnt;
    if (cnt == 0)
      weightList.push_back(opIt->first);
  }
  size_t i = 0;
  while (i < weightList.size()) {
    Op op = weightList[i++];
    weightOps.insert(op);
    const std::set<Edge, EdgeCompare>& outList = newGraph->outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        weightList.push_back(it2->dstOp);
      }
    }
  }
  while (true) {
    bool change = false;
    for (opIt = newGraph->inEdges.begin(); opIt != newGraph->inEdges.end(); opIt++) {
      Op op = opIt->first;
      if (weightOps.find(op) != weightOps.end() && newGraph->num_out_edges(op) == 0) {
        newGraph->remove_node(op);
        change = true;
        break;
      }
    }
    if (!change)
      break;
  }
  return newGraph;
}

Op Graph::find_op_or_fail(size_t guid)
{
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++)
    if (it->first.guid == guid) {
      return it->first;
    }
  assert(false);
}

int Graph::get_operator_list(Op* ops, size_t maxNumOps)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }

  size_t cnt = 0, i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    if ((op.ptr->type == OP_INPUT) || (op.ptr->type == OP_WEIGHT)) {
    } else {
      ops[cnt++] = op;
    }
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
  assert(opList.size() == inEdges.size());
  return cnt;
}

int Graph::get_input_edges(Edge* ops, size_t guid)
{
  Op op = find_op_or_fail(guid);
  assert(inEdges.find(op) != inEdges.end());
  std::set<Edge, EdgeCompare> inList = inEdges[op];
  size_t cnt = inList.size();
  std::set<Edge, EdgeCompare>::const_iterator it2;
  for (it2 = inList.begin(); it2 != inList.end(); it2 ++) {
    Edge e = *it2;
    ops[it2->dstIdx] = e;
  }
  // We manually delete the second input for pool2d
  if (op.ptr->type == OP_POOL2D_MAX || op.ptr->type == OP_POOL2D_AVG) {
    assert(cnt == 2 || cnt == 1);
    cnt = 1;
  }
  return cnt;
}

OpType Graph::get_operator_type(size_t guid)
{
  Op op = find_op_or_fail(guid);
  return op.ptr->type;
}

int Graph::get_operator_int_attr(size_t guid, PMParameter attr)
{
  Op op = find_op_or_fail(guid);
  int ret;
  assert(op.ptr->get_int_parameter(attr, &ret));
  return ret;
}

int Graph::get_num_outputs(size_t guid)
{
  Op op = find_op_or_fail(guid);
  return op.ptr->numOutputs;
}

int Graph::get_input_dims(size_t guid, int* dims, int idx)
{
  Op op = find_op_or_fail(guid);
  assert(op.ptr->numInputs > idx);
  int ndim = op.ptr->inputs[idx].numDim;
  for (int i = 0; i < ndim; i++)
    dims[i] = op.ptr->inputs[idx].dim[i];
  return ndim;
}

void Graph::get_weight_value(size_t guid, DATATYPE* value)
{
  Op op = find_op_or_fail(guid);
  // Assume weight op has one input and one output
  assert(op.ptr->type == OP_WEIGHT);
  assert(op.ptr->numInputs == 1);
  assert(op.ptr->numOutputs == 1);
  assert(op.ptr->inputs[0].data_ptr != NULL);
  model->copy_memory(value, (DATATYPE*) op.ptr->inputs[0].data_ptr,
      sizeof(DATATYPE) * op.ptr->inputs[0].volume());
}

int Graph::get_output_dims(size_t guid, int* dims, int idx)
{
  Op op = find_op_or_fail(guid);
  assert(op.ptr->numOutputs > idx);
  int ndim = op.ptr->outputs[idx].numDim;
  for (int i = 0; i < ndim; i++)
    dims[i] = op.ptr->outputs[idx].dim[i];
  return ndim;
}

int Graph::get_split_lens(size_t guid, int* lens)
{
  Op op = find_op_or_fail(guid);
  assert(op.ptr->type == OP_SPLIT);
  Split* split = (Split*) op.ptr;
  int numSplits = split->numOutputs;
  for (int i = 0; i < numSplits; i++)
    lens[i] = split->outputs[i].dim[split->axis];
  return numSplits;
}

void Graph::add_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx)
{
  assert(dstOp.guid != OP_WEIGHT);
  if (inEdges.find(dstOp) == inEdges.end()) {
    inEdges[dstOp];
  }
  if (outEdges.find(srcOp) == outEdges.end()) {
    outEdges[srcOp];
  }
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  inEdges[dstOp].insert(e);
  outEdges[srcOp].insert(e);
}

void Graph::remove_edge(Edge e)
{
  assert(outEdges[e.srcOp].find(e) != outEdges[e.srcOp].end());
  assert(inEdges[e.dstOp].find(e) != inEdges[e.dstOp].end());
  assert(outEdges[e.srcOp].erase(e) == 1);
  assert(inEdges[e.dstOp].erase(e) == 1);
}

void Graph::replace_node(Op oldOp, Op newOp)
{
  //if (outEdges.find(newOp) == outEdges.end()) {
  //  outEdges[newOp];
  //}
  const std::set<Edge, EdgeCompare>& outSet = outEdges[oldOp];
  std::set<Edge, EdgeCompare>::const_iterator it;
  std::vector<Edge> outList;
  for (it = outSet.begin(); it != outSet.end(); it++)
    outList.push_back(*it);
  for (size_t i = 0; i < outList.size(); i++) {
    Edge e = outList[i];
    remove_edge(e);
    add_edge(newOp, e.dstOp, e.srcIdx, e.dstIdx);
  }
}

void Graph::remove_node(Op oldOp)
{
  assert(outEdges.find(oldOp) != outEdges.end());
  // Asser that it is safe to remove the node
  assert(outEdges[oldOp].size() == 0);
  const std::set<Edge, EdgeCompare>& inSet = inEdges[oldOp];
  std::set<Edge, EdgeCompare>::const_iterator it;
  std::vector<Edge> inList;
  for (it = inSet.begin(); it != inSet.end(); it++)
    inList.push_back(*it);
  for (size_t i = 0; i < inList.size(); i++)
    remove_edge(inList[i]);
  assert(inEdges[oldOp].size() == 0);
  inEdges.erase(oldOp);
  outEdges.erase(oldOp);
}

// We do this in topological order because it will be easier to parse on
// the other end
void Graph::export_to_file(std::string file_name)
{
  ofstream export_fs;
  export_fs.open(file_name.c_str());
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
    {
      opList.push_back(it->first);
    }
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    export_op(export_fs, op);

    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) opList.push_back(it2->dstOp);
    }
  }
  export_fs.close();
  assert(opList.size() == inEdges.size());
}

/* Exports an operator with the following format:
 * guid
 * type
 * dependencies (comma separated list of other ops)
 * parameters (comma separated and type dependent)
 */
void Graph::export_op(ofstream &file_stream, Op &op)
{
  file_stream << op.guid << std::endl;

  file_stream << op.ptr->type << std::endl;

  std::string deps_string;
  std::set<Edge, EdgeCompare> inList = inEdges[op];
  std::set<Edge, EdgeCompare>::const_iterator it;
  int i = 0;
  for (it = inList.begin(); it != inList.end(); it++) {
    deps_string += std::to_string(it->srcOp.guid);
    deps_string += ':';
    deps_string += std::to_string(it->srcIdx);
    deps_string += ',';
    i++;
  }
  if (deps_string.size() > 0)
  {
    deps_string = deps_string.substr(0, deps_string.size()-1);
  }
  file_stream << deps_string.c_str() << std::endl;

  switch (op.ptr->type) {
    case OP_CONV2D:
    { 
      Conv2D* conv = (Conv2D*) op.ptr;
      Tensor t = conv->inputs[0];
      Tensor w = conv->inputs[1];
      int padH, padW;
      conv->get_padding(&padH, &padW);
      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3] << ','; // 3
      file_stream << w.dim[0] << ','; // 4
      file_stream << w.dim[1] << ','; // 5
      file_stream << w.dim[2] << ','; // 6
      file_stream << w.dim[3] << ','; // 7
      file_stream << conv->strideH << ','; // 8
      file_stream << conv->strideW << ','; // 9
      file_stream << conv->padding << ','; // 10
      file_stream << conv->activation << ','; // 11
      file_stream << padH << ','; // 12
      file_stream << padW; // 13
      break;
    }
    case OP_POOL2D_MAX:
    case OP_POOL2D_AVG:
    {
      Pool2D* pool = (Pool2D*) op.ptr;
      Tensor t = pool->inputs[0];
      int padH, padW;
      pool->get_padding(&padH, &padW);
      file_stream << t.dim[0] << ','; // 0
      file_stream << t.dim[1] << ','; // 1
      file_stream << t.dim[2] << ','; // 2
      file_stream << t.dim[3] << ','; // 3
      file_stream << pool->type << ','; // 4
      file_stream << pool->kernelH << ','; // 5
      file_stream << pool->kernelW << ','; // 6
      file_stream << pool->strideH << ','; // 7
      file_stream << pool->strideW << ','; // 8
      file_stream << pool->padding << ','; // 9
      file_stream << pool->activation << ','; // 10
      file_stream << padH << ','; // 11
      file_stream << padW; // 12
      break;
    }
    case OP_SPLIT:
    {
      Split* split = (Split*) op.ptr;
      file_stream << split->axis << ',';
      for (int i = 0; i < split->numOutputs; i++)
      {
        file_stream << split->sizes[i];
        if (i < split->numOutputs - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OP_CONCAT:
    {
      Concat* concat = (Concat*) op.ptr;
      file_stream << concat->axis;
      //TODO: fix below for visualizer
      //Tensor t = concat->inputs[0];
      //file_stream << t.dim[0] << ','; // 0
      //file_stream << t.dim[1] << ','; // 1
      //file_stream << t.dim[2] << ','; // 2
      //file_stream << t.dim[3]; // 3
      break;
    }
    case OP_EW_ADD:
    case OP_EW_MUL:
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    case OP_BATCHNORM:
    case OP_INPUT:
    case OP_WEIGHT:
    {
      Tensor t = op.ptr->inputs[0];
      for (int i = 0; i < t.numDim; i++)
      {
        file_stream << t.dim[i]; // 0 - N
        if (i < t.numDim - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OP_MATMUL: // This doesn't seem to be implemented in run either
    {
      Matmul* matmul = (Matmul*) op.ptr;
      file_stream << matmul->activation << ','; // 0
      file_stream << matmul->outputs[0].numDim; // 1
      break;
    }
    case OP_RESHAPE:
    {
      Reshape *reshape = (Reshape*) op.ptr;
      Tensor t = op.ptr->outputs[0];
      for (int i = 0; i < t.numDim; i++)
      {
        file_stream << t.dim[i]; // 0 - N
        if (i < t.numDim - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    case OP_TRANSPOSE:
    {
      Transpose *transpose = (Transpose*) op.ptr;
      Tensor t = op.ptr->outputs[0];
      int permIdx = transpose->permIdx;
      int ndim = t.numDim, permArray[MAX_DIM];
      for (int i = ndim - 1; i >= 0; i--) {
        permArray[i] = permIdx % ndim;
        permIdx = permIdx / ndim;
      }
      assert(permIdx == 0);
      for (int i = 0; i < ndim; i++) {
        file_stream << t.dim[i];// 0 - N
        if (i < ndim - 1)
        {
          file_stream << ',';
        }
      }
      break;
    }
    default:
      assert(false);
  }
  file_stream << std::endl;
}

size_t Graph::num_in_edges(Op op)
{
  return inEdges[op].size();
}

size_t Graph::num_out_edges(Op op)
{
  return outEdges[op].size();
}

bool Graph::has_edge(Op srcOp, Op dstOp, int srcIdx, int dstIdx)
{
  Edge e(srcOp, dstOp, srcIdx, dstIdx);
  return (inEdges[dstOp].find(e) != inEdges[dstOp].end());
}

size_t Graph::hash(void)
{
  size_t total = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    size_t my = 17 * 31 + (size_t)(it->first.ptr);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      my = my * 31 + std::hash<size_t>()((size_t)(e.srcOp.ptr));
      my = my * 31 + std::hash<int>()(e.srcIdx);
      my = my * 31 + std::hash<int>()(e.dstIdx);
    }
    total += my;
  }
  return total;
}

void Graph::print(void)
{
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first.guid == 0) continue;
    printf("	guid(%zu) type(%d) runtime(%.4lf) op_ptr(%x): ", it->first.guid, it->first.ptr->type, it->first.ptr->runtime, it->first.ptr);
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      printf(" inEdge(guid(%zu) idx(%d))", e.srcOp.guid, e.srcIdx);
    }
    printf("\n");
  }
}

bool Graph::check_correctness(void)
{
  bool okay = true;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = outEdges.begin(); it != outEdges.end(); it++) {
    std::set<Edge, EdgeCompare> list = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = list.begin(); it2 != list.end(); it2++) {
      Edge e = *it2;
      if (!has_edge(e.srcOp, e.dstOp, e.srcIdx, e.dstIdx)) assert(false);
      if (e.srcOp.ptr == NULL) continue;
      Tensor srcTensor = e.srcOp.ptr->outputs[e.srcIdx];
      Tensor dstTensor = e.dstOp.ptr->inputs[e.dstIdx];
      if (srcTensor.numDim != dstTensor.numDim) assert(false);
      for (int i = 0; i < srcTensor.numDim; i++) {
        if (srcTensor.dim[i] != dstTensor.dim[i]) {
          assert(false);
          return false;
        }
        if (srcTensor.stride[i] != dstTensor.stride[i]) {
          //assert(false);
          //return false;
        }
      }
    }
  }
  return okay;
}

float Graph::total_cost(void)
{
  if (totalCost > 0) return totalCost;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  float total = 0.0f;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    if (it->first.ptr != NULL) total += it->first.ptr->runtime;
  }
  totalCost = total;
  return total;
}

bool Graph::has_loop(void)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
  return (opList.size() < inEdges.size());
}

float Graph::run(void)
{
  std::map<Op, int, OpCompare> todos;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  std::vector<Op> opList;
  std::vector<OpBase*> opBaseList;
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    int cnt = 0;
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      if (it2->srcOp.guid > GUID_PRESERVED) cnt ++;
    }
    todos[it->first] = cnt;
    if (todos[it->first] == 0)
      opList.push_back(it->first);
  }
  size_t i = 0;
  while (i < opList.size()) {
    Op op = opList[i++];
    std::set<Edge, EdgeCompare> outList = outEdges[op];
    std::set<Edge, EdgeCompare> inList = inEdges[op];
    std::set<Edge, EdgeCompare>::const_iterator it2;
    assert(inList.size() > 0);
    OpBase* opPtr = NULL;
    // Step 1: prepare inputs
    Tensor inputs[MAX_NUM_INPUTS];
    if ((op.ptr->type == OP_INPUT) || (op.ptr->type == OP_WEIGHT)) {
      assert(inList.size() == 1);
      Edge e = *inList.begin();
      //assert(e.srcOp.ptr == NULL); // NoOp's input must not be any Op
      Tensor t = op.ptr->inputs[0];
      size_t size = sizeof(DATATYPE);
      for (int j = 0; j < t.numDim; j++)
        size *= t.dim[j];
      if (op.ptr->type == OP_INPUT) {
        assert(t.data_ptr == NULL);
        t.data_ptr = (DATATYPE*) model->allocate_memory(size);
      } else {
        assert(t.data_ptr != NULL);
      }
      inputs[0] = t;
    } else {
      for (it2 = inList.begin(); it2 != inList.end(); it2++) {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == it2->srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[it2->dstIdx].data_ptr == NULL); // No duplicated dstIdxes
        inputs[it2->dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
#ifdef DEADCODE
    // Step 1: prepare inputs
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      Edge e = *it2;
      if (e.srcOp.guid == GUID_INPUT) {
        Tensor t = op.ptr->inputs[e.dstIdx];
        t.ptr = (DATATYPE*) model->allocate_memory(sizeof(DATATYPE) * t.size());
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = t;
      } else if (e.srcOp.guid = GUID_WEIGHT) {
        Tensor t = op.ptr->inputs[e.dstIdx];
        t.ptr = (DATATYPE*) model->allocate_memory(sizeof(DATATYPE) * t.size());
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = t;
      } else {
        size_t idx2 = 0;
        for (idx2 = 0; idx2 < opList.size(); idx2++) {
          if (opList[idx2].guid == e.srcOp.guid) break;
        }
        assert(idx2 < i);
        assert(inputs[e.dstIdx].ptr == NULL); // No duplicated dstIdxes
        inputs[e.dstIdx] = opBaseList[idx2]->outputs[it2->srcIdx];
      }
    }
#endif
    // Step 2: create Ops
    switch (op.ptr->type) {
      case OP_CONV2D:
      {
        Conv2D* conv = (Conv2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Conv2D(model, inputs[0], inputs[1],
                           conv->strideH, conv->strideW,
                           conv->padding, conv->activation);
#ifdef USE_CUDNN
        ((Conv2D*)opPtr)->fwdAlgo = conv->fwdAlgo;
#endif
        break;
      }
      case OP_MATMUL:
      {
        Matmul* matmul = (Matmul*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Matmul(model, inputs[0], inputs[1], matmul->activation);
        break;
      }
      case OP_RESHAPE:
      {
        Reshape* reshape = (Reshape*) op.ptr;
        assert(inList.size() == 1);
        std::vector<int> shape;
        for (int i = 0; i < reshape->outputs[0].numDim; i++)
          shape.push_back(reshape->outputs[0].dim[i]);
        opPtr = new Reshape(model, inputs[0], shape);
        break;
      }
      case OP_TRANSPOSE:
      {
        Transpose* transpose = (Transpose*) op.ptr;
        assert(inList.size() == 1);
        int ndim = inputs[0].numDim, permIdx = transpose->permIdx;
        std::vector<int> permVec;
        int permArray[MAX_DIM];
        for (int i = ndim - 1; i >= 0; i--) {
          permArray[i] = permIdx % ndim;
          permIdx = permIdx / ndim;
        }
        assert(permIdx == 0);
        for (int i = 0; i < ndim; i++)
          for (int j = i + 1; j < ndim; j++)
            assert(permArray[i] != permArray[j]);
        for (int i = 0; i < ndim; i++)
          permVec.push_back(permArray[i]);
        opPtr = new Transpose(model, inputs[0], permVec, transpose->shuffle);
        break;
      }
      case OP_EW_ADD:
      case OP_EW_MUL:
      {
        //Element* element = (Element*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Element(model, op.ptr->type, inputs[0], inputs[1]);
        break;
      }
      case OP_ENLARGE:
      {
        //Enlarge* enlarge = (Enlarge*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Enlarge(model, inputs[0], inputs[1]);
        break;
      }
      case OP_MERGE_GCONV:
      {
        MergeGConv* merge = (MergeGConv*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new MergeGConv(model, inputs[0], merge->count);
        break;
      }
      case OP_POOL2D_MAX:
      case OP_POOL2D_AVG:
      {
        Pool2D* pool = (Pool2D*) op.ptr;
        assert(inList.size() == 2);
        opPtr = new Pool2D(model, inputs[0], inputs[1], pool->type,
                           pool->kernelH, pool->kernelW,
                           pool->strideH, pool->strideW,
                           pool->padding, pool->activation);
        break;
      }
      case OP_RELU:
      case OP_SIGMOID:
      case OP_TANH:
      {
        Activation* act = (Activation*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Activation(model, inputs[0], act->type, act->inPlace);
        break;
      }
      case OP_BATCHNORM:
      {
        assert(inList.size() == 5);
        opPtr = new BatchNorm(model, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
        break;
      }
      case OP_SPLIT:
      {
        Split* split = (Split*) op.ptr;
        assert(inList.size() == 1);
        opPtr = new Split(model, inputs[0], split->axis,
                          split->numOutputs, split->sizes);
        break;
      }
      case OP_INPUT:
      case OP_WEIGHT:
      case OP_DROPOUT:
      {
        assert(inList.size() == 1);
        opPtr = new NoOp(model, inputs[0], op.ptr->type);
        break;
      }
      case OP_CONCAT:
      {
        Concat* concat = (Concat*) op.ptr;
        opPtr = new Concat(model, concat->axis, inList.size(), inputs, concat->needCopy);
        break;
      }
      default:
        printf("op.type = %d\n", op.ptr->type);
        assert(false);
    }
    // Step 3: map new Op
    opPtr->map();
    opBaseList.push_back(opPtr);
    for (it2 = outList.begin(); it2 != outList.end(); it2++) {
      todos[it2->dstOp] --;
      //printf("myOp(%zu) dstOp(%zu) dstType(%d) dstTodos(%d)\n",
      //    it2->srcOp.guid, it2->dstOp.guid,
      //    it2->dstOp.ptr->type, todos[it2->dstOp]);
      if (todos[it2->dstOp] == 0) {
        opList.push_back(it2->dstOp);
      }
    }
  }
#ifdef VERBOSE_PRINTS
  for (int i =0; i < opList.size(); i++) {
    printf("opList[%d]: guid(%zu) type(%d)\n", i, opList[i].guid,
           opList[i].ptr->type);
  }
  for (it = inEdges.begin(); it != inEdges.end(); it++) {
    printf("op: guid(%zu) type(%d)\n", it->first.guid, it->first.ptr->type);
    std::set<Edge, EdgeCompare> inList = it->second;
    std::set<Edge, EdgeCompare>::const_iterator it2;
    int cnt = 0;
    for (it2 = inList.begin(); it2 != inList.end(); it2++) {
      printf("    inEdge[%d]: srcOp(%zu) srcIdx(%d) dstOp(%zu) dstIdx(%d)\n", cnt++, it2->srcOp.guid, it2->srcIdx, it2->dstOp.guid, it2->dstIdx);
    }
  }
#endif

  assert(opList.size() == inEdges.size());
  assert(opList.size() == opBaseList.size());

  return model->measure_oplist_runtime(opBaseList);
}

void Graph::print_costs(void)
{
  float exe_time = 0, flops = 0, mem_acc = 0;
  int num_kernels = 0;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
  for (it = inEdges.begin(); it != inEdges.end(); it++)
    it->first.ptr->collect_costs(exe_time, flops, mem_acc, num_kernels);
  printf("        Cost metrics: exe_time(%.4lf) flops(%.4lf) "
         "memory_access(%.4lf) kernel_launches(%d)\n",
         exe_time, flops / 1024.0 / 1024.0 / 1024.0,
         mem_acc * 4.0 / 1024.0 / 1024.0, num_kernels);
}


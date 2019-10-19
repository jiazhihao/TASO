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

#include "taso/substitution.h"
using namespace taso;

GraphXfer* create_avg_pool_conv(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  OpX* avg_pool = subst->create_pool2d_avg(input, weight, 1, 1,
                                           PD_MODE_SAME,
                                           AC_MODE_NONE);
  OpX* conv = subst->create_conv2d(input, weight, 1, 1,
                                   PD_MODE_SAME,
                                   AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(avg_pool->outputs[0], conv->outputs[0]);
  subst->srcOps.push_back(avg_pool);
  subst->dstOps.push_back(conv);
  return subst;
}

GraphXfer* create_two_pools(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX w1 = subst->new_tensor();
  //TensorX w2 = subst->new_tensor();
  OpX* pool1 = subst->create_pool2d_avg(input, w1, 1, 1,
                                        PD_MODE_SAME,
                                        AC_MODE_NONE);
  //OpX* pool2 = subst->create_pool2d_avg(input, w2, 1, 1,
  //                                      PD_MODE_SAME,
  //                                      AC_MODE_NONE);
  //OpX* add = subst->create_element(pool1->outputs[0], pool2->outputs[0],
  //                                 OP_EW_ADD);
  OpX* pool3 = subst->create_conv2d(input, w1, 1, 1,
                                    PD_MODE_SAME,
                                    AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(pool1->outputs[0], pool3->outputs[0]);
  subst->srcOps.push_back(pool1);
  //subst->srcOps.push_back(pool2);
  //subst->srcOps.push_back(add);
  subst->dstOps.push_back(pool3);
  return subst;
}

GraphXfer* GraphXfer::create_conv_relu(Model* model, int strideH, int strideW, PaddingMode mode)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX weight = subst->new_tensor();
  OpX* conv = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                   AC_MODE_NONE);
  OpX* relu = subst->create_activation(conv->outputs[0], OP_RELU);
  OpX* fuse = subst->create_conv2d(input, weight, strideH, strideW, mode,
                                   AC_MODE_RELU, false/*isSrc*/);
  subst->map_output(relu->outputs[0], fuse->outputs[0]);
  subst->srcOps.push_back(conv);
  subst->srcOps.push_back(relu);
  subst->dstOps.push_back(fuse);
  return subst;
}

GraphXfer* GraphXfer::create_enlarge_merge_convs(Model* model, ActiMode activation)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX w1 = subst->new_tensor();
  TensorX w2 = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input, w1, 1, 1, PD_MODE_SAME, activation);
  OpX* conv2 = subst->create_conv2d(input, w2, 1, 1, PD_MODE_SAME, activation);
  subst->srcOps.push_back(conv1);
  subst->srcOps.push_back(conv2);
  OpX* enlarge = subst->create_enlarge(w1, w2, false/*isSrc*/);
  OpX* concat = subst->create_concat(0/*axis*/, 4/*dim*/, enlarge->outputs[0],
                                     w2, false/*isSrc*/);
  OpX* conv3 = subst->create_conv2d(input, concat->outputs[0], 1, 1,
                                    PD_MODE_SAME, activation, false/*isSrc*/);
  OpX* split = subst->create_split(conv3->outputs[0], 1/*axis*/, 2, false/*isSrc*/);
  subst->dstOps.push_back(enlarge);
  subst->dstOps.push_back(concat);
  subst->dstOps.push_back(conv3);
  subst->dstOps.push_back(split);
  subst->map_output(conv1->outputs[0], split->outputs[0]);
  subst->map_output(conv2->outputs[0], split->outputs[1]);
  return subst;
}

GraphXfer* GraphXfer::create_merge_group_convs(Model* model,
                                               int strideH,
                                               int strideW,
                                               ActiMode activation)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input = subst->new_tensor();
  TensorX w = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input, w, strideH, strideW, PD_MODE_SAME, activation);
  subst->srcOps.push_back(conv1);
  OpX* merge = subst->create_merge_gconv(w, 2/*count*/, false/*isSrc*/);
  OpX* conv2 = subst->create_conv2d(input, merge->outputs[0], strideH, strideW, PD_MODE_SAME, activation, false/*isSrc*/);
  subst->dstOps.push_back(merge);
  subst->dstOps.push_back(conv2);
  subst->map_output(conv1->outputs[0], conv2->outputs[0]);
  return subst;
}

GraphXfer* create_merge_seperable_convs(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  TensorX input1 = subst->new_tensor();
  TensorX input2 = subst->new_tensor();
  TensorX w1 = subst->new_tensor();
  TensorX w2 = subst->new_tensor();
  TensorX w3 = subst->new_tensor();
  TensorX w4 = subst->new_tensor();
  OpX* conv1 = subst->create_conv2d(input1, w1, 1, 1, PD_MODE_SAME,
                                    AC_MODE_NONE);
  OpX* conv2 = subst->create_conv2d(input2, w2, 1, 1, PD_MODE_SAME,
                                    AC_MODE_NONE);
  OpX* conv3 = subst->create_conv2d(conv1->outputs[0], w3, 1, 1,
                                    PD_MODE_SAME, AC_MODE_NONE);
  OpX* conv4 = subst->create_conv2d(conv2->outputs[0], w4, 1, 1,
                                    PD_MODE_SAME, AC_MODE_NONE);
  OpX* add = subst->create_element(conv3->outputs[0], conv4->outputs[0],
                                   OP_EW_ADD);
  OpX* concatIn = subst->create_concat(1/*axis*/, 4/*dim*/, input1, input2, false/*isSrc*/);
  OpX* concat1 = subst->create_concat(0/*axis*/, 4/*dim*/, w1, w2, false/*isSrc*/);
  OpX* concat2 = subst->create_concat(1/*axis*/, 4/*dim*/, w3, w4, false/*isSrc*/);
  OpX* conv5 = subst->create_conv2d(concatIn->outputs[0], concat1->outputs[0], 1, 1,
                                    PD_MODE_SAME, AC_MODE_NONE, false/*isSrc*/);
  OpX* conv6 = subst->create_conv2d(conv5->outputs[0], concat2->outputs[0], 1, 1,
                                    PD_MODE_SAME,AC_MODE_NONE, false/*isSrc*/);
  subst->map_output(add->outputs[0], conv6->outputs[0]);
  subst->srcOps.push_back(conv1);
  subst->srcOps.push_back(conv2);
  subst->srcOps.push_back(conv3);
  subst->srcOps.push_back(conv4);
  subst->srcOps.push_back(add);
  subst->dstOps.push_back(concatIn);
  subst->dstOps.push_back(concat1);
  subst->dstOps.push_back(concat2);
  subst->dstOps.push_back(conv5);
  subst->dstOps.push_back(conv6);
  return subst;
}

bool get_parameter_from_pb(const GraphSubst::Operator& pbOp,
                           PMParameter pm,
                           int &value)
{
  for (int i = 0; i < pbOp.para_size(); i++)
    if (pbOp.para(i).key() == pm) {
      value = pbOp.para(i).value();
      return true;
    }
  return false;  
}

void GraphXfer::create_operator_from_pb(const GraphSubst::Operator& pbOp,
                                        std::map<int, TensorX>& mappedInputs,
                                        bool isSrcOp)
{
  // Step 1: create inputs
  TensorX inputs[MAX_NUM_INPUTS];
  assert(pbOp.input_size() <= MAX_NUM_INPUTS);
  for (int i = 0; i < pbOp.input_size(); i++) {
    const GraphSubst::Tensor& tensor = pbOp.input(i);
    if (tensor.opid() < 0) {
      int opId = tensor.opid();
      if (mappedInputs.find(opId) == mappedInputs.end()) {
        mappedInputs[opId] = new_tensor();
        assert(isSrcOp); // assert we are still in the src graph
      }
      inputs[i] = mappedInputs[opId];
    } else {
      int opId = tensor.opid();
      int tsId = tensor.tsid();
      if (isSrcOp)
        inputs[i] = srcOps[opId]->outputs[tsId];
      else
        inputs[i] = dstOps[opId]->outputs[tsId];
    }
  }
  // Step 2: create op
  OpType type = (OpType) pbOp.type();
  OpX* opx = NULL;
  switch (type) {
    case OP_CONV2D:
    {
      assert(pbOp.input_size() == 2);
      int strideH, strideW, padding, activation;
      //get_parameter_from_pb(pbOp, PM_KERNEL_H, kernelH);
      //get_parameter_from_pb(pbOp, PM_KERNEL_W, kernelW);
      assert(get_parameter_from_pb(pbOp, PM_STRIDE_H, strideH));
      assert(get_parameter_from_pb(pbOp, PM_STRIDE_W, strideW));
      assert(get_parameter_from_pb(pbOp, PM_PAD, padding));
      assert(get_parameter_from_pb(pbOp, PM_ACTI, activation));
      opx = create_conv2d(inputs[0], inputs[1], strideH, strideW,
          (PaddingMode) padding, (ActiMode) activation, isSrcOp);
      break;
    }
    case OP_CONCAT:
    {
      int numDim, axis;
      assert(get_parameter_from_pb(pbOp, PM_AXIS, axis));
      assert(get_parameter_from_pb(pbOp, PM_NUMDIM, numDim));
      opx = create_concat(axis, numDim, pbOp.input_size(), inputs, isSrcOp);
      break;
    }
    case OP_EW_ADD:
    case OP_EW_MUL:
    {
      assert(pbOp.input_size() == 2);
      opx = create_element(inputs[0], inputs[1], type, isSrcOp);
      break;
    }
    case OP_SPLIT:
    {
      assert(pbOp.input_size() == 1);
      int numOutputs, axis;
      assert(get_parameter_from_pb(pbOp, PM_AXIS, axis));
      assert(get_parameter_from_pb(pbOp, PM_NUM_OUTPUTS, numOutputs));
      opx = create_split(inputs[0], axis, numOutputs, isSrcOp);
      break;
    }
    case OP_RELU:
    case OP_SIGMOID:
    case OP_TANH:
    {
      assert(pbOp.input_size() == 1);
      opx = create_activation(inputs[0], type);
      break;
    }
    case OP_MUL:
    {
      assert(pbOp.input_size() == 2);
      opx = create_mul(inputs[0], inputs[1]);
      break;
    }
    case OP_ENLARGE:
    {
      assert(pbOp.input_size() == 2);
      //int kernelH, kernelW;
      //assert(get_parameter_from_pb(pbOp, PM_KERNEL_H, kernelH));
      //assert(get_parameter_from_pb(pbOp, PM_KERNEL_W, kernelW));
      opx = create_enlarge(inputs[0], inputs[1], isSrcOp);
      break;
    }
    case OP_MATMUL:
    {
      assert(pbOp.input_size() == 2);
      int activation;
      assert(get_parameter_from_pb(pbOp, PM_ACTI, activation));
      opx = create_matmul(inputs[0], inputs[1], (ActiMode) activation);
      break;
    }
    case OP_TRANSPOSE:
    {
      assert(pbOp.input_size() == 1);
      int numDim, permIdx, perm[MAX_DIM], shuffle;
      assert(get_parameter_from_pb(pbOp, PM_NUMDIM, numDim));
      assert(get_parameter_from_pb(pbOp, PM_PERM, permIdx));
      assert(get_parameter_from_pb(pbOp, PM_OUTSHUFFLE, shuffle));
      for (int i = numDim-1; i >=0; i--) {
        perm[i] = permIdx % numDim;
        permIdx = permIdx / numDim;
      }
      assert(permIdx == 0);
      for (int i = 0; i < numDim; i++)
        for (int j = i + 1; j < numDim; j++)
          assert(perm[i] != perm[j]);
      opx = create_transpose(inputs[0], numDim, perm, shuffle);
      break;
    }
    case OP_POOL2D_MAX:
    case OP_POOL2D_AVG:
    case OP_BATCHNORM:
    default:
    {
      assert(false);
    }
  }
  assert(opx != NULL);
  if (isSrcOp)
    srcOps.push_back(opx);
  else
    dstOps.push_back(opx);
}

void GraphXfer::load_graph_xfer_from_pb_file(Model* model,
                                             std::vector<GraphXfer*>& xfers,
                                             std::string filename)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  GraphSubst::RuleCollection collection;
  std::fstream input(filename, ios::in);
  assert(collection.ParseFromIstream(&input));
  //printf("Number of generated substitutions = %d\n", collection.rule_size());
  for (int i = 0; i < collection.rule_size(); i++) {
    const GraphSubst::Rule& rule = collection.rule(i);
    std::map<int, TensorX> mappedInputs;
    GraphXfer* subst = new GraphXfer(model);
    for (int j = 0; j < rule.srcop_size(); j++)
      subst->create_operator_from_pb(rule.srcop(j), mappedInputs, true);
    for (int j = 0; j < rule.dstop_size(); j++)
      subst->create_operator_from_pb(rule.dstop(j), mappedInputs, false);
    for (int j = 0; j < rule.mappedoutput_size(); j++) {
      const GraphSubst::MapOutput& mapOutput = rule.mappedoutput(j);
      int srcOpId = mapOutput.srcopid();
      int dstOpId = mapOutput.dstopid();
      int srcTsId = mapOutput.srctsid();
      int dstTsId = mapOutput.dsttsid();
      assert(srcOpId < (int)subst->srcOps.size());
      assert(dstOpId < (int)subst->dstOps.size());
      assert(srcTsId < (int)subst->srcOps[srcOpId]->outputs.size());
      assert(dstTsId < (int)subst->dstOps[dstOpId]->outputs.size());
      subst->map_output(subst->srcOps[srcOpId]->outputs[srcTsId],
                        subst->dstOps[dstOpId]->outputs[dstTsId]);
    }
    xfers.push_back(subst);
  }
}

// Helper functions
TNParameter to_tn_parameter(bool isInput, int n)
{
  switch (n) {
    case 0: return isInput ? IN_0 : OU_0;
    case 1: return isInput ? IN_1 : OU_1;
    case 2: return isInput ? IN_2 : OU_2;
    case 3: return isInput ? IN_3 : OU_3;
    case 4: return isInput ? IN_4 : OU_4;
    case 5: return isInput ? IN_5 : OU_5;
    default:
      assert(false);
  }
  assert(false);
}

DIMParameter to_dim_parameter(int n)
{
  switch (n) {
    case 0: return DIM_0;
    case 1: return DIM_1;
    case 2: return DIM_2;
    case 3: return DIM_3;
    default:
      assert(false);
  }
  assert(false);
}

PMConstraint::PMConstraint(Compare c, PMParameter p, int v)
: comp(c), para(p), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p, DIMParameter d, int v)
: singlePara(true), comp(c), para1(p), dim1(d), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p1, DIMParameter d1,
                           TNParameter p2, DIMParameter d2)
: singlePara(false), comp(c), para1(p1), para2(p2), dim1(d1), dim2(d2) {}

Tensor TensorX::to_tensor(const GraphXfer* xfer) const
{
  if (op != NULL) {
    assert(op->mapOp.ptr != NULL);
    return op->mapOp.ptr->outputs[idx];
  } else {
    std::multimap<int, std::pair<Op, int> >::const_iterator it;
    it = xfer->mappedInputs.find(idx);
    assert(it != xfer->mappedInputs.end());
    Op op = it->second.first;
    int outIdx = it->second.second;
    return op.ptr->outputs[outIdx];
  }
}

//void add_out_edges(TensorX e)
//{
//  if (e.op != NULL) e.op->numOutEdges ++;
//}

OpX::OpX(OpType _type, TensorX in1, int numOutputs)
: type(_type)
{
  inputs.push_back(in1);
  switch (type) {
    case OP_RESHAPE:
    case OP_TRANSPOSE:
    case OP_RELU:
    case OP_TANH:
    case OP_SIGMOID:
    case OP_MERGE_GCONV:
    {
      TensorX out(this, 0);
      outputs.push_back(out);
      break;
    }
    case OP_SPLIT:
      for (int i = 0; i < numOutputs; i++) {
        TensorX out(this, i);
        outputs.push_back(out);
      }
      break;
    default:
      assert(false);
  }
}

OpX::OpX(OpType _type, TensorX in1, TensorX in2)
: type(_type)
{
  inputs.push_back(in1);
  inputs.push_back(in2);
  TensorX out(this, 0);
  switch (type) {
    case OP_CONV2D:
    case OP_EW_ADD:
    case OP_EW_MUL:
    case OP_POOL2D_AVG:
    case OP_CONCAT:
    case OP_MATMUL:
    case OP_MUL:
    case OP_ENLARGE:
      outputs.push_back(out);
      break;
    default:
      assert(false);
  }
}

OpX::OpX(OpType _type, int n, TensorX* ins)
: type(_type)
{
  for (int i = 0; i < n; i++) {
    inputs.push_back(ins[i]);
  }
  TensorX out(this, 0);
  outputs.push_back(out);
}

bool OpX::add_pm_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint pmc(comp, para, value);
  pmConstraints.push_back(pmc);
  return true;
}

bool OpX::add_input_constraint(Compare comp, TNParameter para,
                               DIMParameter dim, int value)
{
  TNConstraint tnc(comp, para, dim, value);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::add_input_constraint(Compare comp,
                               TNParameter para1, DIMParameter dim1,
                               TNParameter para2, DIMParameter dim2)
{
  TNConstraint tnc(comp, para1, dim1, para2, dim2);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::get_pm_constraint(PMParameter para, int& value) const
{
  for (size_t i = 0; i < pmConstraints.size(); i++)
    if ((pmConstraints[i].comp == COMPARE_EQ)
    && (pmConstraints[i].para == para)) {
      value = pmConstraints[i].value;
      return true;
    }
  return false;
}

bool SrcOp::add_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint ooc(comp, para, value);
  constraints.push_back(ooc);
  return true;
}

bool SrcOp::match(Op op)
{
  if (op.guid == 0) return false;
  if (type != OP_ANY && type != op.ptr->type)
    return false;
  bool pass = true;
  for (size_t i = 0; i < constraints.size(); i++) {
    PMConstraint ooc = constraints[i];
    int actValue = 0;
    assert(op.ptr->get_int_parameter(ooc.para, &actValue));
    switch (ooc.comp) {
      case COMPARE_EQ:
        if (actValue != ooc.value) pass = false;
        break;
      case COMPARE_NE:
        if (actValue == ooc.value) pass = false;
        break;
      case COMPARE_LT:
        if (actValue >= ooc.value) pass = false;
        break;
      case COMPARE_LE:
        if (actValue > ooc.value) pass = false;
        break;
      case COMPARE_GT:
        if (actValue <= ooc.value) pass = false;
        break;
      case COMPARE_GE:
        if (actValue < ooc.value) pass = false;
        break;
      default:
        assert(false);
    }
  }
  return pass;
}

/*
SrcEdge::SrcEdge(int _idx, SrcOp* _op)
: idx(_idx), op(_op)
{}

DstEdge::DstEdge(int _idx, DstOp* _op)
: idx(_idx), op(_op)
{}
*/

GraphXfer::GraphXfer(Model* _model)
: model(_model), tensorId(10)
{}

OpX* GraphXfer::create_activation(TensorX input, OpType type, bool isSrcOp)
{
  OpX* activation = new OpX(type, input);
  return activation;
}

OpX* GraphXfer::create_conv2d(TensorX input, TensorX weight,
                              //int kernelH, int kernelW,
                              int strideH, int strideW,
                              PaddingMode padding,
                              ActiMode activation,
                              bool isSrcOp)
{
  OpX* conv = new OpX(OP_CONV2D, input, weight);
  //conv->add_pm_constraint(COMPARE_EQ, PM_KERNEL_H, kernelH);
  //conv->add_pm_constraint(COMPARE_EQ, PM_KERNEL_W, kernelW);
  conv->add_pm_constraint(COMPARE_EQ, PM_STRIDE_H, strideH);
  conv->add_pm_constraint(COMPARE_EQ, PM_STRIDE_W, strideW);
  conv->add_pm_constraint(COMPARE_EQ, PM_PAD, padding);
  conv->add_pm_constraint(COMPARE_EQ, PM_ACTI, activation);
  //conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_2, kernelH);
  //conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_3, kernelW);
  // The following is no longer true because of group conv
  //conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_1, IN_0, DIM_1);
  return conv;
}

OpX* GraphXfer::create_element(TensorX input0, TensorX input1,
                               OpType type, bool isSrcOp)
{
  OpX* element = new OpX(type, input0, input1);
  return element;
}

OpX* GraphXfer::create_pool2d_avg(TensorX input, TensorX weight,
                                  int strideH, int strideW,
                                  PaddingMode padding,
                                  ActiMode activation,
                                  bool isSrcOp)
{
  OpX* pool = new OpX(OP_POOL2D_AVG, input, weight);
  pool->add_pm_constraint(COMPARE_EQ, PM_STRIDE_H, strideH);
  pool->add_pm_constraint(COMPARE_EQ, PM_STRIDE_W, strideW);
  pool->add_pm_constraint(COMPARE_EQ, PM_PAD, padding);
  pool->add_pm_constraint(COMPARE_EQ, PM_ACTI, activation);
  pool->add_input_constraint(COMPARE_EQ, IN_1, DIM_0, IN_0, DIM_1);
  return pool;
}

OpX* GraphXfer::create_matmul(TensorX input, TensorX weight,
                              ActiMode activation,
                              bool isSrcOp)
{
  OpX* matmul = new OpX(OP_MATMUL, input, weight);
  matmul->add_pm_constraint(COMPARE_EQ, PM_ACTI, activation);
  matmul->add_input_constraint(COMPARE_EQ, IN_1, DIM_0, IN_0, DIM_1);
  return matmul;
}

OpX* GraphXfer::create_mul(TensorX x, TensorX y, bool isSrcOp)
{
  OpX* mul = new OpX(OP_MUL, x, y);
  mul->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, 0);
  return mul;
}

OpX* GraphXfer::create_transpose(TensorX input, int numDim, int* perm,
                                 int shuffle)
{
  OpX* transpose = new OpX(OP_TRANSPOSE, input);
  int permIdx = 0;
  for (int i = 0; i < numDim; i++)
    permIdx = permIdx * numDim + perm[i];
  transpose->add_pm_constraint(COMPARE_EQ, PM_PERM, permIdx);
  transpose->add_pm_constraint(COMPARE_EQ, PM_OUTSHUFFLE, shuffle);
  transpose->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, numDim);
  return transpose;
}

OpX* GraphXfer::create_enlarge(TensorX w1, TensorX w2, bool isSrcOp)
{
  OpX* enlarge = new OpX(OP_ENLARGE, w1, w2);
  //enlarge->add_pm_constraint(COMPARE_EQ, PM_KERNEL_H, kernelH);
  //enlarge->add_pm_constraint(COMPARE_EQ, PM_KERNEL_W, kernelW);
  enlarge->add_input_constraint(COMPARE_LE, IN_0, DIM_2, IN_1, DIM_2);
  enlarge->add_input_constraint(COMPARE_LE, IN_0, DIM_3, IN_1, DIM_3);
  return enlarge;
}

OpX* GraphXfer::create_merge_gconv(TensorX w, int count, bool isSrcOp)
{
  OpX* merge = new OpX(OP_MERGE_GCONV, w);
  merge->add_pm_constraint(COMPARE_EQ, PM_MERGE_GCONV_COUNT, count);
  return merge;
}

OpX* GraphXfer::create_concat(int axis, int numDim, TensorX in1, TensorX in2, bool isSrcOp)
{
  TensorX ins[2];
  ins[0] = in1; ins[1] = in2;
  return create_concat(axis, numDim, 2, ins, isSrcOp);
}

OpX* GraphXfer::create_concat(int axis, int numDim, int n, TensorX* ins, bool isSrcOp)
{
  OpX* concat = new OpX(OP_CONCAT, n, ins);
  concat->add_pm_constraint(COMPARE_EQ, PM_AXIS, axis);
  concat->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, numDim);
  for (int i = 1; i < n; i++) {
    TNParameter in_i = to_tn_parameter(true/*is_input*/, i);
    concat->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND,
                                 in_i, DIM_ND);
    for (int j = 0; j < numDim; j++) {
      DIMParameter dim_j = to_dim_parameter(j);
      if (j != axis)
        concat->add_input_constraint(COMPARE_EQ, IN_0, dim_j,
                                     in_i, dim_j);
    }
  }
  return concat;
}

OpX* GraphXfer::create_split(TensorX input, int axis, int n, bool isSrcOp)
{
  OpX* split = new OpX(OP_SPLIT, input, n);
  split->add_pm_constraint(COMPARE_EQ, PM_AXIS, axis);
  return split;
}

TensorX GraphXfer::new_tensor(void)
{
  TensorX t;
  t.op = NULL;
  t.idx = tensorId++;
  return t;
}

bool GraphXfer::map_output(TensorX src, TensorX dst)
{
  mappedOutputs[src] = dst;
  return true;
}

//void GraphXfer::add_src_op(SrcOp* src)
//{
//  srcInEdges[src];
//  srcOutEdges[src];
//  srcOps.push_back(src);
//}
//
//void GraphXfer::add_dst_op(DstOp* dst)
//{
//  dstInEdges[dst];
//  dstOutEdges[dst];
//  dstOps.push_back(dst);
//}

//void GraphXfer::add_src_edge(SrcOp* srcOp, SrcOp* dstOp, int srcIdx, int dstIdx)
//{
//  SubEdge<SrcOp> e(srcOp, dstOp, srcIdx, dstIdx);
//  srcInEdges[dstOp].insert(e);
//  srcOutEdges[srcOp].insert(e);
//}

//void GraphXfer::add_dst_edge(DstOp* srcOp, DstOp* dstOp, int srcIdx, int dstIdx)
//{
//  SubEdge<DstOp> e(srcOp, dstOp, srcIdx, dstIdx);
//  dstInEdges[dstOp].insert(e);
//  dstOutEdges[srcOp].insert(e);
//}

//bool GraphXfer::add_constraint(Compare comp,
//                               SrcOp* src, PMParameter srcPara,
//                               SrcOp* dst, PMParameter dstPara)
//{
//  TwoOpConstraint gc(comp, src, srcPara, dst, dstPara);
//  constraints.push_back(gc);
//  return true;
//}

//bool GraphXfer::map_input(SrcOp* src, DstOp* dst)
//{
//  assert(src->mapInput == NULL);
//  assert(dst->mapInput == NULL);
//  src->mapInput = dst;
//  dst->mapInput = src;
//  return true;
//}

//bool GraphXfer::map_output(SrcOp* src, DstOp* dst)
//{
//  assert(src->mapOutput == NULL);
//  assert(dst->mapOutput == NULL);
//  src->mapOutput = dst;
//  dst->mapOutput = src;
//  return true;
//}

bool GraphXfer::can_match(OpX* srcOp, Op op, Graph* graph)
{
  if (srcOp->type != op.ptr->type) return false;
  // check num input tensors
  if ((int)srcOp->inputs.size() != op.ptr->numInputs) return false;
  // check pmConstraints
  for (size_t i = 0; i < srcOp->pmConstraints.size(); i++) {
    PMConstraint pmc = srcOp->pmConstraints[i];
    int actValue = 0;
    assert(op.ptr->get_int_parameter(pmc.para, &actValue));
    //printf("pmc[%d] para(%d) comp(%d) value(%d) actValue(%d)\n",
    //       i, pmc.para, pmc.comp, pmc.value, actValue);
    switch (pmc.comp) {
      case COMPARE_EQ:
      {
        if (actValue != pmc.value) return false;
        break;
      }
      case COMPARE_NE:
      {
        if (actValue == pmc.value) return false;
        break;
      }
      case COMPARE_LT:
      {
        if (actValue >= pmc.value) return false;
        break;
      }
      case COMPARE_LE:
      {
        if (actValue > pmc.value) return false;
        break;
      }
      case COMPARE_GT:
      {
        if (actValue <= pmc.value) return false;
        break;
      }
      case COMPARE_GE:
      {
        if (actValue < pmc.value) return false;
        break;
      }
      default:
        assert(false);
    }
  }
  // check inputs
  std::map<int, std::pair<Op, int> > newMapInputs;
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // input tensor
      std::multimap<int, std::pair<Op, int> >::const_iterator it;
      it = mappedInputs.find(in.idx);
      if (it != mappedInputs.end()) {
        Op mappedOp = it->second.first;
        int mappedIdx = it->second.second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
          return false;
      } else {
        std::map<int, std::pair<Op, int> >::const_iterator newit;
        newit = newMapInputs.find(in.idx);
        if (newit != newMapInputs.end()) {
          Op mappedOp = newit->second.first;
          int mappedIdx = newit->second.second;
          if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
            return false;
        } else {
          std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
          std::set<Edge, EdgeCompare>::const_iterator it2;
          for (it2 = list.begin(); it2 != list.end(); it2++) {
            Edge e = *it2;
            if (e.dstIdx == (int)i) {
              newMapInputs.insert(std::make_pair(in.idx,
                                      std::make_pair(e.srcOp, e.srcIdx)));
            }
          }
        }
        // Do nothing when we check the match
        /* mapped in.idx to an op
        std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
        std::set<Edge, EdgeCompare>::const_iterator it2;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          Edge e = *it2;
          if (e.dstIdx == i)
            mappedInputs[in.idx] = std::make_pair(e.srcOp, e.srcIdx);
        }*/
      }
    } else {
      // intermediate tensor
      assert(in.op->mapOp.ptr != NULL);
      if (!(graph->has_edge(in.op->mapOp, op, in.idx, i)))
        return false;
    }
  }
  // check tnConstraints
  for (size_t i = 0; i < srcOp->tnConstraints.size(); i++) {
    TNConstraint tnc = srcOp->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_input_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
      {
        if (actValue != expValue) return false;
        break;
      }
      case COMPARE_NE:
      {
        if (actValue == expValue) return false;
        break;
      }
      case COMPARE_LT:
      {
        if (actValue >= expValue) return false;
        break;
      }
      case COMPARE_LE:
      {
        if (actValue > expValue) return false;
        break;
      }
      case COMPARE_GT:
      {
        if (actValue <= expValue) return false;
        break;
      }
      case COMPARE_GE:
      {
        if (actValue < expValue) return false;
        break;
      }
      default:
        assert(false);
    }
  }
  return true;
}

void GraphXfer::match(OpX* srcOp, Op op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputs
      std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
      std::set<Edge, EdgeCompare>::const_iterator it2;
      for (it2 = list.begin(); it2 != list.end(); it2++) {
        Edge e = *it2;
        if (e.dstIdx == (int)i) {
          mappedInputs.insert(std::make_pair(in.idx,
                                  std::make_pair(e.srcOp, e.srcIdx)));
        }
      }
    }
  }
  // Map srcOp to Op
  srcOp->mapOp = op;
  mappedOps[op] = srcOp;
}

void GraphXfer::unmatch(OpX* srcOp, Op op, Graph* graph)
{
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // Update mappedInputsa
      std::multimap<int, std::pair<Op, int> >::iterator it;
      it = mappedInputs.find(in.idx);
      mappedInputs.erase(it);
    }
  }
  // Unmap op
  mappedOps.erase(op);
  srcOp->mapOp.guid = 0;
  srcOp->mapOp.ptr = NULL;
}

void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::set<size_t>& hashmap, float threshold, int maxNumOps)
{
  //printf("run: depth(%d) srcOps.size(%zu) graph.size(%zu) candidates(%zu)\n", depth, srcOps.size(), graph->inEdges.size(), candidates.size());
  if (depth >= (int)srcOps.size()) {
    // Create dst operators
    bool pass = true;
    std::vector<OpX*>::const_iterator dstIt;
    for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt++)
      if (pass) {
        OpX* dstOp = *dstIt;
        pass = (pass & create_new_operator(dstOp, dstOp->mapOp));
      }
    if (!pass) return;
    // Check that output tensors with external edges are mapped
    std::map<Op, OpX*, OpCompare>::const_iterator opIt;
    for (opIt = mappedOps.begin(); opIt != mappedOps.end(); opIt++) {
      const std::set<Edge, EdgeCompare>& list = graph->outEdges[opIt->first];
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mappedOps.find(it->dstOp) == mappedOps.end()) {
          // dstOp is external, (srcOp, srcIdx) must be in mappedOutputs
          TensorX srcTen;
          srcTen.op = opIt->second;
          srcTen.idx = it->srcIdx;
          if (mappedOutputs.find(srcTen) == mappedOutputs.end()) {
            pass = false;
            return;
          }
        }
    }
    // Generate a new graph by applying xfer rule
    Graph* newGraph = create_new_graph(graph);
    // Check that the new graph should not have any loop
    if (newGraph->has_loop()) {
      //printf("Found a new graph with LOOP!!!!\n");
      delete newGraph;
      return;
    }
    // TODO: remove me for better performance
    assert(newGraph->check_correctness());
    if (newGraph->total_cost() < threshold && (int)newGraph->inEdges.size() < maxNumOps) {
      if (hashmap.find(newGraph->hash()) == hashmap.end()) {
        hashmap.insert(newGraph->hash());
        candidates.push(newGraph);
      }
    } else {
      delete newGraph;
    }
  } else {
    OpX* srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      //printf("can_match(%d)\n", can_match(srcOp, it->first, graph));
      if (can_match(srcOp, it->first, graph)
      && (mappedOps.find(it->first) == mappedOps.end())) {
        Op op = it->first;
        // Check mapOutput
        match(srcOp, op, graph);
        run(depth + 1, graph, candidates, hashmap, threshold, maxNumOps);
        unmatch(srcOp, op, graph);
      }
    }
  }
}

Graph* GraphXfer::create_new_graph(Graph* graph)
{
  Graph* newGraph = new Graph();
  // Step 1: map dst ops
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  std::vector<OpX*>::const_iterator dstIt;
  // Step 2: add edges to the graph
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mappedOps.find(opIt->first) == mappedOps.end()) {
      // Unmapped ops
      const std::set<Edge, EdgeCompare>& list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mappedOps.find(it->srcOp) != mappedOps.end()) {
          // mapped src -> unmapped dst
          TensorX srcTen;
          srcTen.op = mappedOps[it->srcOp];
          srcTen.idx = it->srcIdx;
          assert(mappedOutputs.find(srcTen) != mappedOutputs.end());
          TensorX dstTen = mappedOutputs[srcTen];
          newGraph->add_edge(dstTen.op->mapOp, it->dstOp, dstTen.idx, it->dstIdx);
        } else {
          // unmapped src -> unmmaped dst
          newGraph->add_edge(it->srcOp, it->dstOp, it->srcIdx, it->dstIdx);
        }
    }
  // Step 3: add edges for mapped ops
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    OpX* dstOp = *dstIt;
    for (size_t i = 0; i < dstOp->inputs.size(); i++)
      if (dstOp->inputs[i].op == NULL) {
        // unmapped src -> mapped dst
        std::multimap<int, std::pair<Op, int> >::const_iterator it
            = mappedInputs.find(dstOp->inputs[i].idx);
        assert(it != mappedInputs.end());
        std::pair<Op, int> srcEdge = it->second;
        newGraph->add_edge(srcEdge.first, dstOp->mapOp, srcEdge.second, i);
      } else {
        // mapped src -> mapped dst
        OpX* srcOp = dstOp->inputs[i].op;
        int srcIdx = dstOp->inputs[i].idx;
        newGraph->add_edge(srcOp->mapOp, dstOp->mapOp, srcIdx, i);
      }
  }
  return newGraph;
}

bool GraphXfer::create_new_operator(const OpX* opx, Op& op)
{
  switch (opx->type) {
    case OP_CONV2D:
    {
      assert(opx->inputs.size() == 2);
      Tensor input = opx->inputs[0].to_tensor(this);
      Tensor weight = opx->inputs[1].to_tensor(this);
      int strideH, strideW, padding, activation;
      assert(opx->get_pm_constraint(PM_STRIDE_H, strideH));
      assert(opx->get_pm_constraint(PM_STRIDE_W, strideW));
      assert(opx->get_pm_constraint(PM_PAD, padding));
      assert(opx->get_pm_constraint(PM_ACTI, activation));
      op = model->get_or_create_conv2d(input, weight, strideH, strideW,
                                       (PaddingMode)padding,
                                       (ActiMode)activation);
      break;
    }
    case OP_EW_ADD:
    case OP_EW_MUL:
    {
      assert(opx->inputs.size() == 2);
      Tensor input0 = opx->inputs[0].to_tensor(this);
      Tensor input1 = opx->inputs[1].to_tensor(this);
      op = model->get_or_create_element(opx->type, input0, input1);
      break;
    }
    case OP_MATMUL:
    {
      assert(opx->inputs.size() == 2);
      Tensor input = opx->inputs[0].to_tensor(this);
      Tensor weight = opx->inputs[1].to_tensor(this);
      int activation;
      assert(opx->get_pm_constraint(PM_ACTI, activation));
      op = model->get_or_create_matmul(input, weight,
                                       (ActiMode)activation);
      break;
    }
    case OP_TRANSPOSE:
    {
      assert(opx->inputs.size() == 1);
      Tensor input = opx->inputs[0].to_tensor(this);
      int permIdx, shuffle;
      assert(opx->get_pm_constraint(PM_PERM, permIdx));
      assert(opx->get_pm_constraint(PM_OUTSHUFFLE, shuffle));
      op = model->get_or_create_transpose(input, permIdx, (bool)shuffle);
      break;
    }
    case OP_ENLARGE:
    {
      assert(opx->inputs.size() == 2);
      Tensor w1 = opx->inputs[0].to_tensor(this);
      Tensor w2 = opx->inputs[1].to_tensor(this);
      //int kernelH, kernelW;
      //assert(opx->get_pm_constraint(PM_KERNEL_H, kernelH));
      //assert(opx->get_pm_constraint(PM_KERNEL_W, kernelW));
      op = model->get_or_create_enlarge(w1, w2);
      break;
    }
    case OP_MERGE_GCONV:
    {
      assert(opx->inputs.size() == 1);
      Tensor weight = opx->inputs[0].to_tensor(this);
      int count;
      assert(opx->get_pm_constraint(PM_MERGE_GCONV_COUNT, count));
      op = model->get_or_create_merge_gconv(weight, count);
      break;
    }
    case OP_CONCAT:
    {
      // TODO: assume don't need copy for now
      Tensor inputs[MAX_NUM_INPUTS];
      bool needCopy[MAX_NUM_INPUTS];
      for (size_t i = 0; i < opx->inputs.size(); i++) {
        inputs[i] = opx->inputs[i].to_tensor(this);
        needCopy[i] = false;
      }
      int axis;
      assert(opx->get_pm_constraint(PM_AXIS, axis));
      op = model->get_or_create_concat(axis, opx->inputs.size(), inputs, needCopy);
      break;
    }
    case OP_SPLIT:
    {
      int axis;
      Tensor input = opx->inputs[0].to_tensor(this);
      assert(opx->get_pm_constraint(PM_AXIS, axis));
      op = model->get_or_create_split(input, axis, opx->outputs.size());
      break;
    }
    case OP_RELU:
    case OP_TANH:
    case OP_SIGMOID:
    {
      assert(opx->inputs.size() == 1);
      Tensor input = opx->inputs[0].to_tensor(this);
      op = model->get_or_create_activation(input, opx->type, true);
      break;
    }
    default:
    {
      printf("opx->type = %d\n", opx->type);
      assert(false);
    }
  }
  // Check operator validness
  if (op == Op::INVALID_OP)
    return false;
  // Check tnConstraints
  for (size_t i = 0; i < opx->tnConstraints.size(); i++) {
    TNConstraint tnc = opx->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_input_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
        if (actValue != expValue) return false;
        break;
      case COMPARE_NE:
        if (actValue == expValue) return false;
        break;
      case COMPARE_LT:
        if (actValue >= expValue) return false;
        break;
      case COMPARE_LE:
        if (actValue > expValue) return false;
        break;
      case COMPARE_GT:
        if (actValue <= expValue) return false;
        break;
      case COMPARE_GE:
        if (actValue < expValue) return false;
        break;
      default:
        assert(false);
    }
  }
  return true;
}

/*
void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::set<size_t>& hashmap, float threshold)
{
  if (depth >= srcOps.size()) {
    // Check two op constraints
    bool pass = true;
    for (size_t i = 0; i < constraints.size(); i++) {
      TwoOpConstraint toc = constraints[i];
      int value1, value2;
      assert(toc.op1->mapOp.ptr != NULL);
      assert(toc.op2->mapOp.ptr != NULL);
      assert(toc.op1->mapOp.ptr->get_parameter(toc.para1, &value1));
      assert(toc.op2->mapOp.ptr->get_parameter(toc.para2, &value2));
      switch (toc.comp) {
        case COMPARE_EQ:
          if (value1 != value2) pass = false;
          break;
        case COMPARE_NE:
          if (value1 == value2) pass = false;
          break;
        case COMPARE_LT:
          if (value1 >= value2) pass = false;
          break;
        case COMPARE_GT:
          if (value1 <= value2) pass = false;
          break;
        default:
          assert(false);
      }
    }
    // Generate a new graph by applying xfer rule
    if (pass) {
      Graph* newGraph = create_new_graph(graph);
      //assert(newGraph->check_correctness());
      if (newGraph->total_cost() < threshold) {
        if (hashmap.find(newGraph->hash()) == hashmap.end()) {
          hashmap.insert(newGraph->hash());
          candidates.push(newGraph);
        }
      } else {
        delete newGraph;
      }
    }
  } else {
    // Match srcOps[depth];
    SrcOp* srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      if (srcOp->match(it->first)
      && (mapped.find(it->first) == mapped.end())) {
        Op op = it->first;
        std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> > list = srcInEdges[srcOp];
        std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> >::const_iterator it2;
        // Check edges in the source subgraph
        bool pass = true;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          SubEdge<SrcOp> edge = *it2;
          if (!graph->has_edge(edge.srcOp->mapOp, op, edge.srcIdx, edge.dstIdx)) pass = false;
        }
        // Check mapInput/mapOutput
        bool extraInputs = false, extraOutputs = false;
        if (srcInEdges[srcOp].size() != graph->num_in_edges(op))
          extraInputs = true;
        if (srcOutEdges[srcOp].size() != graph->num_out_edges(op))
          extraOutputs = true;
        if (!srcOp->mapInput && extraInputs)
          pass = false;
        if (!srcOp->mapOutput && extraOutputs)
          pass = false;
        // Serch for the next op if pass the check
        if (pass) {
          srcOp->mapOp = op;
          mapped.insert(op);
          run(depth + 1, graph, candidates, hashmap, threshold);
          mapped.erase(op);
          srcOp->mapOp.guid = 0;
          srcOp->mapOp.ptr = NULL;
        }
      }
    }
  }
}

Graph* GraphXfer::create_new_graph(Graph* graph)
{
  Graph* newGraph = new Graph(graph->model);
  // Step 1: add operators to the graph
  std::vector<DstOp*>::iterator dstIt;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mapped.find(opIt->first) == mapped.end()) {
      newGraph->inEdges[opIt->first];
      newGraph->outEdges[opIt->first];
    }
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    DstOp* dstOp = *dstIt;
    dstOp->mapOp = dstOp->create_operator(graph->model);
    newGraph->inEdges[dstOp->mapOp];
    newGraph->outEdges[dstOp->mapOp];
  }
  // Step 2: add edges to the graph
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mapped.find(opIt->first) != mapped.end()) {
      // Mapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->srcOp) != mapped.end()) {
          // mapped src -> mapped dst
          // Do nothing!
        } else {
          // unmapped src -> mapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == opIt->first.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapInput != NULL);
          Op op = srcOps[i]->mapInput->mapOp;
          Edge e(it->srcOp, op, it->srcIdx, it->dstIdx);
          newGraph->inEdges[op].insert(e);
          newGraph->outEdges[it->srcOp].insert(e);
        }
    } else {
      // Unmapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->srcOp) != mapped.end()) {
          // mapped src -> unmapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == it->srcOp.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapOutput != NULL);
          Op op = srcOps[i]->mapOutput->mapOp;
          Edge e(op, opIt->first, it->srcIdx, it->dstIdx);
          newGraph->inEdges[opIt->first].insert(e);
          newGraph->outEdges[op].insert(e);
        } else {
          // unmapped src -> unmapped dst
          Edge e(it->srcOp, opIt->first, it->srcIdx, it->dstIdx);
          newGraph->inEdges[opIt->first].insert(e);
          newGraph->outEdges[it->srcOp].insert(e);
        }
    }
  // Step 3: add edges in the dstInEdges
  std::map<DstOp*, std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > >::iterator dstOpIt;
  for (dstOpIt = dstInEdges.begin(); dstOpIt != dstInEdges.end(); dstOpIt++) {
    std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > list = dstOpIt->second;
    std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> >::const_iterator it;
    for (it = list.begin(); it != list.end(); it++) {
      Op src = it->srcOp->mapOp, dst = dstOpIt->first->mapOp;
      Edge e(src, dst, it->srcIdx, it->dstIdx);
      newGraph->inEdges[dst].insert(e);
      newGraph->outEdges[src].insert(e);
    }
  }
  return newGraph;
}
*/

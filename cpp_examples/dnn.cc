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
#include "example_utils.h"
#include "bert.h"
#include "nasnet_a.h"
#include "nasrnn.h"
#include "resnet50.h"
#include "resnext50.h"
#include <cstring>

using namespace taso;

enum DNNModel {
  None,
  BERT,
  NASNETA,
  NASRNN,
  Resnet50,
  Resnext50,
};

DNNModel name2model(std::string name) {
  if (name == "bert") return BERT;
  if (name == "nasnet-a") return NASNETA;
  if (name == "nasrnn") return NASRNN;
  if (name == "resnet50") return Resnet50;
  if (name == "resnext50") return Resnext50;
  return None;
}

void parse_args(float& alpha,
                int& budget,
                std::string& exportFileName,
                DNNModel& dnnModel,
                int argc, char **argv) {
  alpha = 1.05;
  budget = 300;
  exportFileName = "";
  dnnModel = None;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--alpha")) {
      alpha = std::atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--budget")) {
      budget = std::atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--export")) {
      exportFileName = argv[++i];
      continue;
    }
    if (!strcmp(argv[i], "--dnn")) {
      dnnModel = name2model(std::string(argv[++i]));
      continue;
    }
    fprintf(stderr, "Found unknown option!!\n");
    assert(false);
  }
  if (dnnModel == None) fprintf(stderr, "Must specify a DNN model!\n");
}

int main(int argc, char **argv) {
  int budget;
  float alpha;
  std::string exportFileName;
  DNNModel dnn = None;
  parse_args(alpha, budget, exportFileName, dnn, argc, argv);
  printf("DNN Model %d, alpha = %.4lf, budget = %d\n", dnn, alpha, budget);

  Graph* graph = nullptr;
  switch (dnn) {
    case BERT:
      graph = bert(alpha, budget);
      break;
    case NASNETA:
      graph = nasnet_a(alpha, budget);
      break;
    case NASRNN:
      graph = nasrnn(alpha, budget);
      break;
    case Resnet50:
      graph = resnet50(alpha, budget);
      break;
    case Resnext50:
      graph = resnext50(alpha, budget);
      break;
    default:
      assert(false);
  }
  if (!exportFileName.empty()) graph->export_to_file(exportFileName);
  return 0;
}

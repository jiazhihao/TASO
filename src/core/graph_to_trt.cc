#ifdef TRT
#include <algorithm>
#include <chrono>
#include "xflow/ops.h"
#include "xflow/cuda_helper.h"

#define TIMING_ITERATIONS 10

class SplitPlugin : public IPlugin {
public:
  SplitPlugin(int nOuts, int *channels_, int axis): nOuts(nOuts), axis(axis) {
    assert(nOuts <= MAX_NUM_OUTPUTS);
    for (int i = 0; i < nOuts; i++) {
      channels[i] = channels_[i];
    }
  }

  int getNbOutputs() const override {
    return nOuts;
  }

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override {
    assert(nbInputDims == 1);
    assert(inputs[0].nbDims == 3);
    int outChannelsSum = 0;
    for (int i = 0; i < nOuts; i++) {
      outChannelsSum += channels[i];
    }
    assert(inputs[0].d[axis] == outChannelsSum);
    return Dims3{axis == 0 ? channels[index] : inputs[0].d[0],
                 axis == 1 ? channels[index] : inputs[0].d[1],
                 axis == 2 ? channels[index] : inputs[0].d[2]};
  }

  void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    int maxBatchSize) override {

    assert(maxBatchSize == 1);
    d = inputDims[0];
  }

  int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace,
    cudaStream_t stream) override {

    /*
    TODO replace this -- example for axis == 0
    int runningChannels = 0;
    for (int i = 0; i < nOuts; i++) {
      auto inputSlice = static_cast<const uint32_t*>(inputs[0]) + runningChannels * d[1] * d[2];
      checkCUDA(cudaMemcpyAsync(outputs[i], inputSlice, sizeof(uint32_t) * channels[i] * d[1] * d[2],
        cudaMemcpyDeviceToDevice, stream));
      runningChannels += channels[i];
    }
    */
    return 0;
  }

  int initialize() override { return 0; }
  void terminate() override {}
  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
  size_t getSerializationSize() override { return 0; }
  void serialize(void *buffer) override {}
private:
  int nOuts, axis;
  Dims d;
  int channels[MAX_NUM_OUTPUTS];
};

class Logger : public ILogger
{
public:

    Logger(): Logger(Severity::kWARNING) {}

    Logger(Severity severity): reportableSeverity(severity) {}

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};
};

static Logger gLogger;

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS); // %-400.400s
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

} gProfiler;

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    checkCUDA(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

void Graph::buildTRTNetworkHelper(INetworkDefinition *network, std::map<SrcEdge, ITensor *, SrcEdgeCompare>& outputs, Edge fullEdge) {
  SrcEdge edge(fullEdge.srcIdx, fullEdge.srcOp);
  if (outputs.find(edge) != outputs.end()) {
    return;
  }

  std::set<Edge, EdgeCompare> inList = inEdges[edge.op];
  std::vector<ITensor *> inputs;
  for (auto it = inList.begin(); it != inList.end(); it++) {
    if (it->srcOp.guid > GUID_PRESERVED) {
      buildTRTNetworkHelper(network, outputs, *it);
      inputs.push_back(outputs[(SrcEdge) {it->srcIdx, it->srcOp}]);
    } else if (it->srcOp.guid == GUID_WEIGHT) {
      assert(edge.op.ptr->type == OpBase::OP_NOOP);
      return;
    }
  }
  if (inputs.size() == 0) {
    Tensor input = edge.op.ptr->inputs[0];
    Dims d;
    d.nbDims = input.numDim;
    for (int i = 0; i < d.nbDims; i++) {
      d.d[i] = input.dim[i];
    }
    char name[255];
    sprintf(name, "in%zd", edge.op.guid);
    ITensor *trt_input = network->addInput(name, DataType::kFLOAT, d);
    outputs[edge] = trt_input;
    return;
  }

  switch (edge.op.ptr->type) {
    case OpBase::OP_CONV2D:
    {
      assert(inputs.size() == 2);
      assert(inputs[0]->getDimensions().nbDims == 4);
      Conv2D* conv = (Conv2D*) edge.op.ptr;
      int inputC = inputs[0]->getDimensions().d[1];
      int kernelH, kernelW, outputC, padH, padW;
      kernelH = conv->inputs[1].dim[2];
      kernelW = conv->inputs[1].dim[3];
      outputC = conv->inputs[1].dim[0];
      assert(inputC % conv->inputs[1].dim[1] == 0);
      int groups = inputC / conv->inputs[1].dim[1];
      conv->get_padding(&padH, &padW);
      int numWeights = kernelH * kernelW * outputC * conv->inputs[1].dim[1];
      auto trt_conv = network->addConvolution(*inputs[0], outputC, DimsHW{kernelH, kernelW},
        (Weights) {DataType::kFLOAT, malloc(sizeof(uint32_t) * numWeights), numWeights}, // TODO memory leak
        (Weights) {DataType::kFLOAT, nullptr, 0});
      char name[255];
      sprintf(name, "conv%zd:%dx%d/%dx%d/%d/%d",
        edge.op.guid, kernelH, kernelW, conv->strideH, conv->strideW, inputC, outputC);
      trt_conv->setNbGroups(groups);
      trt_conv->setName(name);
      trt_conv->setStride(DimsHW{conv->strideH, conv->strideW});
      trt_conv->setPadding(DimsHW{padH, padW});
      outputs[edge] = trt_conv->getOutput(0);
      break;
    }
    case OpBase::OP_POOL2D_MAX:
    case OpBase::OP_POOL2D_AVG:
    {
      assert(inputs.size() == 2);
      Pool2D* pool = (Pool2D*) edge.op.ptr;
      int padH, padW;
      pool->get_padding(&padH, &padW);
      auto trt_pool = network->addPooling(*inputs[0],
        pool->type == OpBase::OP_POOL2D_MAX ? PoolingType::kMAX : PoolingType::kAVERAGE,
        DimsHW{pool->kernelH, pool->kernelW});
      trt_pool->setStride(DimsHW{pool->strideH, pool->strideW});
      trt_pool->setPadding(DimsHW{padH, padW});
      outputs[edge] = trt_pool->getOutput(0);
      break;
    }
    case OpBase::OP_RELU:
    {
      assert(inputs.size() == 1);
      outputs[edge] = network->addActivation(*inputs[0], ActivationType::kRELU)->getOutput(0);
      break;
    }
    case OpBase::OP_TANH:
    {
      assert(inputs.size() == 1);
      outputs[edge] = network->addActivation(*inputs[0], ActivationType::kTANH)->getOutput(0);
      break;
    }
    case OpBase::OP_SIGMOID:
    {
      assert(inputs.size() == 1);
      outputs[edge] = network->addActivation(*inputs[0], ActivationType::kSIGMOID)->getOutput(0);
      break;
    }
    case OpBase::OP_BATCHNORM:
    {
      assert(inputs.size() == 1);
      float scale_param = 5.0f;
      float shift_param = 1.0f;
      outputs[edge] = network->addScale(*inputs[0], ScaleMode::kUNIFORM,
        (Weights) {DataType::kFLOAT, &shift_param, 1}, (Weights) {DataType::kFLOAT, &scale_param, 1},
        (Weights) {DataType::kFLOAT, nullptr, 0})->getOutput(0);
      break;
    }
    case OpBase::OP_SPLIT:
    {
      assert(inputs.size() == 1);
      Split *split = (Split *) edge.op.ptr;
      /*
      SplitPlugin *trt_split_plugin = new SplitPlugin(edge.op.ptr->numOutputs, split->sizes, split->axis); // TODO memory leak
      auto trt_split_layer = network->addPlugin(&inputs[0], 1, *trt_split_plugin);
      for (int i = 0; i < trt_split_layer->getNbOutputs(); i++) {
        outputs[(SrcEdge) {i, edge.op}] = trt_split_layer->getOutput(i);
      }
      */
      Dims startD;
      startD.nbDims = split->inputs[0].numDim;
      for (int i = 0; i < startD.nbDims; i++) {
        startD.d[i] = 0;
      }
      Dims sizeD;
      sizeD.nbDims = split->inputs[0].numDim;
      for (int i = 0; i < sizeD.nbDims; i++) {
        sizeD.d[i] = split->inputs[0].dim[i];
      }
      Dims strideD;
      strideD.nbDims = split->inputs[0].numDim;
      for (int i = 0; i < strideD.nbDims; i++) {
        strideD.d[i] = 1;
      }
      int cumDim = 0;
      for (int i = 0; i < split->numOutputs; i++) {
        startD.d[split->axis] = cumDim;
        sizeD.d[split->axis] = split->sizes[i];
        outputs[(SrcEdge) {i, edge.op}] =
          network->addSlice(*inputs[0], startD, sizeD, strideD)->getOutput(0);
        cumDim += split->sizes[i];
      }
      break;
    }
    case OpBase::OP_RESHAPE:
    {
      assert(inputs.size() == 1);
      auto trt_reshape = network->addShuffle(*inputs[0]);
      Dims new_dims;
      new_dims.nbDims = edge.op.ptr->outputs[0].numDim;
      for (int i = 0; i < new_dims.nbDims; i++) {
        new_dims.d[i] = edge.op.ptr->outputs[0].dim[i];
      }
      trt_reshape->setReshapeDimensions(new_dims);
      outputs[edge] = trt_reshape->getOutput(0);
      break;
    }
    case OpBase::OP_EW_ADD:
    case OpBase::OP_EW_MUL:
    {
      assert(inputs.size() == 2);
      outputs[edge] = network->addElementWise(*inputs[0], *inputs[1],
        edge.op.ptr->type == OpBase::OP_EW_ADD ? ElementWiseOperation::kSUM : ElementWiseOperation::kPROD)->getOutput(0);
      break;
    }
    case OpBase::OP_MATMUL:
    {
      assert(inputs.size() == 2);
      Matmul *matmul = (Matmul *) edge.op.ptr;
      Dims weight_dims;
      weight_dims.nbDims = matmul->inputs[1].numDim;
      for (int i = 0; i < weight_dims.nbDims; i++) {
        weight_dims.d[i] = matmul->inputs[1].dim[i];
      }
      char name[255];
      sprintf(name, "matmul%zd_weights", edge.op.guid);
      ITensor *trt_weight_matrix = network->addInput(name, DataType::kFLOAT, weight_dims);
      auto trt_mm = network->addMatrixMultiply(*inputs[0], false, *trt_weight_matrix, false);
      if (matmul->activation != OpBase::AC_MODE_NONE) {
        ActivationType at = matmul->activation == OpBase::AC_MODE_RELU ? ActivationType::kRELU :
          matmul->activation == OpBase::AC_MODE_SIGMOID ? ActivationType::kSIGMOID : ActivationType::kTANH;
        outputs[edge] = network->addActivation(*trt_mm->getOutput(0), at)->getOutput(0);
      } else {
        outputs[edge] = trt_mm->getOutput(0);
      }
      break;
    }
    case OpBase::OP_NOOP:
    {
      assert(inputs.size() == 1);
      outputs[edge] = inputs[0];
      break;
    }
    case OpBase::OP_CONCAT:
    {
      assert(inputs.size() > 1);
      Concat *concat = (Concat *) edge.op.ptr;
      // Concat weights: directly return
      if (inputs[0] == NULL)
        return;
      int nd = inputs[0]->getDimensions().nbDims;
      for (int i = 0; i < inputs.size(); i++) {
        assert(inputs[i]->getDimensions().nbDims == nd);
        for (int j = 0; j < nd; j++)
          if (j !=concat->axis) {
            assert(inputs[i]->getDimensions().d[j] == inputs[0]->getDimensions().d[j]);
          }
      }
      auto trt_concat = network->addConcatenation(&inputs[0], inputs.size());
      trt_concat->setAxis(concat->axis);
      outputs[edge] = trt_concat->getOutput(0);
      break;
    }
    default:
      assert(false);
  }
}

void Graph::buildTRTNetwork(INetworkDefinition *network) {
  std::map<SrcEdge, ITensor *, SrcEdgeCompare> outputs;
  for (auto it = inEdges.begin(); it != inEdges.end(); it++) {
    if (outEdges.find(it->first) != outEdges.end()) {
      if (outEdges[it->first].size() > 0)
        continue;
    }
    //if (outEdges.find(it->first) == outEdges.end()) {
      assert(it->first.ptr->numOutputs == 1);
      Edge outEdge(it->first, it->first, 0, 0); // will immediately be converted to SrcEdge
      buildTRTNetworkHelper(network, outputs, outEdge);
      network->markOutput(*outputs[(SrcEdge) {0, it->first}]);
    //}
  }
}

void runGraphTRT(Graph *graph) {
  IBuilder* builder = createInferBuilder(gLogger);
  INetworkDefinition* network = builder->createNetwork();
  graph->buildTRTNetwork(network);
  IRuntime* runtime = createInferRuntime(gLogger);

  builder->setMaxBatchSize(1);
  builder->setMaxWorkspaceSize(1 << 30);

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  network->destroy();
  builder->destroy();

  IExecutionContext* context = engine->createExecutionContext();
  context->setProfiler(&gProfiler);
  int batchSize = 1;

  int nbBindings = engine->getNbBindings();
  //assert(nbBindings == 2);

  std::vector<void*> buffers(nbBindings);

  for (int i = 0; i < nbBindings; ++i) {
    Dims dims = engine->getBindingDimensions(i);
    //assert(dims.nbDims == 3);
    int64_t v = 1;
    for (int j = 0; j < dims.nbDims; j++) {
      v *= dims.d[j];
    }
    buffers[i] = safeCudaMalloc(sizeof(uint32_t) * v);
  }

  int numberRun = TIMING_ITERATIONS;
  float total = 0, ms;
  for (int run = 0; run < numberRun; run++) {
      auto t_start = std::chrono::high_resolution_clock::now();
      context->execute(batchSize, &buffers[0]);
      auto t_end = std::chrono::high_resolution_clock::now();
      ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
      total += ms;
  }

  total /= numberRun;
  std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;

  for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx) {
    checkCUDA(cudaFree(buffers[bindingIdx]));
  }

  context->destroy();
  engine->destroy();
  gProfiler.printLayerTimes();
}

#endif

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors: Raimondas Galvelis, Raul P. Pelaez                           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "HipTorchKernels.h"
#include "HipTorchKernelSources.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <hip/hip_runtime_api.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>

using namespace TorchPlugin;
using namespace OpenMM;
using namespace std;

// macro for checking the result of synchronization operation on HIP
// copied from `openmm/platforms/cuda/src/HipParallelKernels.cpp`
#define CHECK_RESULT(result, prefix)                                             \
    if (result != hipSuccess) {                                                \
        std::stringstream m;                                                     \
        m << prefix << ": " << hipGetErrorString((hipError_t)result) << " (" << result << ")"\
          << " at " << __FILE__ << ":" << __LINE__;                              \
        throw OpenMMException(m.str());                                          \
    }

HipCalcTorchForceKernel::HipCalcTorchForceKernel(string name, const Platform& platform, HipContext& cu) : CalcTorchForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    // Explicitly activate the primary context
    CHECK_RESULT(hipDevicePrimaryCtxRetain(&primaryContext, cu.getDeviceIndex()), "Failed to retain the primary context");
}

HipCalcTorchForceKernel::~HipCalcTorchForceKernel() {
    hipDevicePrimaryCtxRelease(cu.getDeviceIndex());
}

void HipCalcTorchForceKernel::initialize(const System& system, const TorchForce& force, torch::jit::script::Module& module) {
    this->module = module;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    outputsForces = force.getOutputsForces();
    for (int i = 0; i < force.getNumGlobalParameters(); i++)
        globalNames.push_back(force.getGlobalParameterName(i));
    for (int i = 0; i < force.getNumEnergyParameterDerivatives(); i++) {
        paramDerivs.insert(force.getEnergyParameterDerivativeName(i));
        cu.addEnergyParameterDerivative(force.getEnergyParameterDerivativeName(i));
    }
    int numParticles = system.getNumParticles();

    // Push the PyTorch context
    // NOTE: Pytorch is always using the primary context.
    //       It makes the primary context current, if it is not a case.
    CHECK_RESULT(hipCtxPushCurrent(primaryContext), "Failed to push the HIP context");

    // Initialize HIP objects for PyTorch
    const torch::Device device(torch::kCUDA, cu.getDeviceIndex()); // This implicitly initialize PyTorch (kCUDA maps to HIP on ROCm builds)
    this->module.to(device);
    this->module.eval();
    this->module = torch::jit::freeze(this->module);
    torch::TensorOptions options = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(!outputsForces));
    boxTensor = torch::empty({3, 3}, options);
    energyTensor = torch::empty({0}, options);
    forceTensor = torch::empty({0}, options);
    for (const string& name : globalNames)
        globalTensors[name] = torch::tensor({0}, options);
    // Pop the PyToch context
    hipCtx_t ctx;
    CHECK_RESULT(hipCtxPopCurrent(&ctx), "Failed to pop the HIP context");
    assert(primaryContext == ctx); // Check that PyTorch haven't messed up the context stack

    // Initialize HIP objects for OpenMM-Torch
    ContextSelector selector(cu); // Switch to the OpenMM context
    map<string, string> defines;
    hipModule_t program = cu.createModule(HipTorchKernelSources::torchForce, defines);
    copyInputsKernel = cu.getKernel(program, "copyInputs");
    addForcesKernel = cu.getKernel(program, "addForces");
    auto properties = force.getProperties();
    const std::string useCUDAGraphsString = properties["useCUDAGraphs"];
    if (useCUDAGraphsString == "true")
        useGraphs = true;
    else if (useCUDAGraphsString == "false" || useCUDAGraphsString == "")
        useGraphs = false;
    else
        throw OpenMMException("TorchForce: invalid value of \"useCUDAGraphs\"");
    if (useGraphs) {
        this->warmupSteps = std::stoi(properties["CUDAGraphWarmupSteps"]);
        if (this->warmupSteps <= 0) {
            throw OpenMMException("TorchForce: \"CUDAGraphWarmupSteps\" must be a positive integer");
        }
    }
}

/**
 * Get a pointer to the data in a PyTorch tensor.
 * The tensor is converted to the correct data type if necessary.
 */
static void* getTensorPointer(OpenMM::HipContext& cu, torch::Tensor& tensor) {
    void* data;
    if (cu.getUseDoublePrecision()) {
        data = tensor.to(torch::kFloat64).data_ptr<double>();
    } else {
        data = tensor.to(torch::kFloat32).data_ptr<float>();
    }
    return data;
}

/**
 * Prepare the inputs for the PyTorch model, copying positions from the OpenMM context.
 */
void HipCalcTorchForceKernel::prepareTorchInputs(ContextImpl& context, vector<torch::jit::IValue>& inputs, map<string, torch::Tensor>& globalTensors) {
    int numParticles = cu.getNumAtoms();
    // Get pointers to the atomic positions and simulation box
    void* posData = getTensorPointer(cu, posTensor);
    void* boxData = getTensorPointer(cu, boxTensor);
    // Copy the atomic positions and simulation box to PyTorch tensors
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        void* inputArgs[] = {&posData,
                             &boxData,
                             &cu.getPosq().getDevicePointer(),
                             &cu.getAtomIndexArray().getDevicePointer(),
                             &numParticles,
                             cu.getPeriodicBoxVecXPointer(),
                             cu.getPeriodicBoxVecYPointer(),
                             cu.getPeriodicBoxVecZPointer()};
        cu.executeKernel(copyInputsKernel, inputArgs, numParticles);
        CHECK_RESULT(hipDeviceSynchronize(), "Failed to synchronize the HIP context"); // Synchronize before switching to the PyTorch context
    }
    // Prepare the input of the PyTorch model
    inputs = {posTensor};
    if (usePeriodic)
        inputs.push_back(boxTensor);
    for (const string& name : globalNames) {
        // PyTorch requires us to set requires_grad to false before initializing a tensor.
        globalTensors[name].set_requires_grad(false);
        globalTensors[name][0] = context.getParameter(name);
        if (paramDerivs.find(name) != paramDerivs.end())
            globalTensors[name].set_requires_grad(true);
        inputs.push_back(globalTensors[name]);
    }
}

/**
 * Add the computed forces to the total atomic forces.
 */
void HipCalcTorchForceKernel::addForces(torch::Tensor& forceTensor) {
    int numParticles = cu.getNumAtoms();
    // Get a pointer to the computed forces
    void* forceData = getTensorPointer(cu, forceTensor);
    CHECK_RESULT(hipDeviceSynchronize(), "Failed to synchronize the HIP context"); // Synchronize before switching to the OpenMM context
    // Add the computed forces to the total atomic forces
    {
        ContextSelector selector(cu); // Switch to the OpenMM context
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        int forceSign = (outputsForces ? 1 : -1);
        void* forceArgs[] = {&forceData, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &forceSign};
        cu.executeKernel(addForcesKernel, forceArgs, numParticles);
        CHECK_RESULT(hipDeviceSynchronize(), "Failed to synchronize the HIP context"); // Synchronize before switching to the PyTorch context
    }
}

/**
 * This function launches the workload in a way compatible with HIP
 * graphs as far as OpenMM-Torch goes.  Capturing this function when
 * the model is not itself graph compatible (due to, for instance,
 * implicit synchronizations) will result in a HIP error.
 */
static void executeGraph(bool outputsForces, bool includeForces, torch::jit::script::Module& module, vector<torch::jit::IValue>& inputs, torch::Tensor& posTensor, torch::Tensor& energyTensor,
                         torch::Tensor& forceTensor, map<string, torch::Tensor>& globalTensors, set<string> paramDerivs) {
    vector<torch::Tensor> gradInputs;
    if (!outputsForces && includeForces)
        gradInputs.push_back(posTensor);
    for (auto& name : paramDerivs)
        gradInputs.push_back(globalTensors[name]);
    auto none = torch::Tensor();
    if (outputsForces) {
        auto outputs = module.forward(inputs).toTuple();
        energyTensor = outputs->elements()[0].toTensor();
        forceTensor = outputs->elements()[1].toTensor();
        if (gradInputs.size() > 0)
            energyTensor.backward(none, false, false, gradInputs);
    } else {
        energyTensor = module.forward(inputs).toTensor();
        // Compute force by backpropagating the PyTorch model
        // HIP graph capture sometimes fails if backwards is not explicitly requested w.r.t positions
        // See https://github.com/openmm/openmm-torch/pull/120/
        if (gradInputs.size() > 0)
            energyTensor.backward(none, false, false, gradInputs);
        if (includeForces) {
            // This is minus the forces, we change the sign later on
            forceTensor = posTensor.grad().clone();
            // Zero the gradient to avoid accumulating it
            posTensor.grad().zero_();
        }
    }
}

double HipCalcTorchForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Push to the PyTorch context
    CHECK_RESULT(hipCtxPushCurrent(primaryContext), "Failed to push the HIP context");
    vector<torch::jit::IValue> inputs;
    prepareTorchInputs(context, inputs, globalTensors);
    if (!useGraphs) {
        executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
    } else {
        // Record graph if not already done
        bool is_graph_captured = false;
        if (graphs.find(includeForces) == graphs.end()) {
	    //HIP graph capture must occur in a non-default stream
            const auto stream = c10::hip::getStreamFromPool(false, cu.getDeviceIndex());
	        const c10::StreamGuard guard(stream);
            // Warmup the graph workload before capturing.  This first
            // run  before  capture sets  up  allocations  so that  no
            // allocations are  needed after.  Pytorch's  allocator is
            // stream  capture-aware and,  after warmup,  will provide
            // record static pointers and shapes during capture.
            try {
                for (int i = 0; i < this->warmupSteps; i++)
                    executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
            }
            catch (std::exception& e) {
                throw OpenMMException(string("TorchForce Failed to warmup the model before graph construction. Torch reported the following error:\n") + e.what());
            }
            graphs[includeForces].capture_begin();
            try {
                executeGraph(outputsForces, includeForces, module, inputs, posTensor, energyTensor, forceTensor, globalTensors, paramDerivs);
                is_graph_captured = true;
                graphs[includeForces].capture_end();
            }
            catch (std::exception& e) {
                if (!is_graph_captured) {
                    graphs[includeForces].capture_end();
                }
                throw OpenMMException(string("TorchForce Failed to capture the model into a HIP graph. Torch reported the following error:\n") + e.what());
            }
            for (const string& name : paramDerivs)
                globalTensors[name].grad().zero_();
        }
        // Use the same stream as the OpenMM context, even if it is the default stream
        const auto openmmStream = cu.getCurrentStream();
        const auto stream = c10::hip::getStreamFromExternal(openmmStream, cu.getDeviceIndex());
        const c10::StreamGuard guard(stream);
        graphs[includeForces].replay();
    }
    if (includeForces) {
        addForces(forceTensor);
    }
    map<string, double>& energyParamDerivs = cu.getEnergyParamDerivWorkspace();
    for (const string& name : paramDerivs) {
        energyParamDerivs[name] += globalTensors[name].grad().item<double>();
        globalTensors[name].grad().zero_();
    }
    // Get energy
    const double energy = energyTensor.item<double>(); // This implicitly synchronizes the PyTorch context
    // Pop to the PyTorch context
    hipCtx_t ctx;
    CHECK_RESULT(hipCtxPopCurrent(&ctx), "Failed to pop the HIP context");
    assert(primaryContext == ctx); // Check that the correct context was popped
    return energy;
}

HipCalcPythonTorchForceKernel::HipCalcPythonTorchForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& contextImpl, OpenMM::ComputeContext& cc) :
        CommonCalcPythonTorchForceKernel(name, platform, contextImpl, cc), cu(dynamic_cast<HipContext&>(cc)) {
}

void HipCalcPythonTorchForceKernel::initialize(const ContextImpl& context, const PythonTorchForce& force) {
    CommonCalcPythonTorchForceKernel::initialize(context, force);
    torch::Device device(torch::kCUDA, cu.getDeviceIndex());
    torch::TensorOptions options = torch::TensorOptions().device(device).dtype(cu.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32);
    posTensor = torch::empty({numParticles, 3}, options.requires_grad(true));
}

torch::Tensor HipCalcPythonTorchForceKernel::getPositions() {
    ContextSelector selector(cc);
    copyPositionsKernel->execute(numParticles);
    void* source = cu.unwrap(positionsArray).getDevicePointer();
    void* dest = (void*) getTensorPointer(cu, posTensor);
    CHECK_RESULT(hipMemcpyDtoDAsync(dest, source, positionsArray.getSize()*positionsArray.getElementSize(), cu.getCurrentStream()), "Error copying positions");
    return posTensor;
}

void HipCalcPythonTorchForceKernel::addForces(torch::Tensor forceTensor) {
    ContextSelector selector(cc);
    forceTensor = forceTensor.to(posTensor.device()).to(posTensor.dtype());
    void* source = (void*) getTensorPointer(cu, forceTensor);
    void* dest = cu.unwrap(forcesArray).getDevicePointer();
    CHECK_RESULT(hipMemcpyDtoDAsync(dest, source, forcesArray.getSize()*forcesArray.getElementSize(), cu.getCurrentStream()), "Error copying forces");
    addForcesKernel->execute(cc.getNumAtoms());
}

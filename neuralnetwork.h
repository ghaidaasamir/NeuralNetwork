#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>
#include <ctime>
#include <random> 
#include <algorithm>

__global__ void computeFilterGradientsKernel(float *input, float *output_grad, float *filter_grad,
    int inputWidth, int inputHeight, int filterWidth, int filterHeight, int outputWidth, int outputHeight);
__global__ void computeWeightGradientsKernel(float *d_input, float *d_output_grad, float *d_weight_grad, 
    int outputSize, int inputSize, int batchSize);
__global__ void computeBiasGradientsKernel(float *output_grad, float *bias_grad, int outputSize, int batchSize);
__global__ void computeInputGradientsKernel(float *weights, float *output_gradient, float *input_gradient, int inputSize, int outputSize, int batchSize);
__global__ void updateWeightsKernel(float *weights, float *weight_grad, float learningRate, int weightSize);
__global__ void updateBiasesKernel(float *biases, float *bias_grad, float learningRate, int biasSize);
__global__ void computeLossKernel(float* output, float* target, float* loss, int n);
__global__ void clipGradientsKernel(float* gradients, int totalElements, float threshold);

__global__ void forwardPassKernel(float *d_input, float *d_weights, float *d_biases, float *d_output, 
                                  int inputSize, int outputSize, int batchSize, bool last_layer);
__global__ void computeGradientsKernel(float *outputs, float *next_layer_grads, float *current_layer_grads, float *next_layer_weights, int current_output_size, int next_output_size, int batchSize);
__global__ void computeGradientsKernel_last(float *outputs, float *targets, float *grads, int outputSize, int batchSize, bool isOutputLayer);

class NeuralNetwork {
    
public:

    std::vector<int> layer_input_sizes;
    std::vector<int> layer_output_sizes;

    std::vector<float*> d_weights; 
    std::vector<float*> d_biases; 
    
    std::vector<float*> d_output_grad, d_weight_grad, d_bias_grad, d_input_grad;
    std::vector<float*> d_layer_inputs, d_layer_outputs;

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_target = nullptr;
    std::vector<int> layer_sizes;
    float learningRate;
    int batchSize;

    NeuralNetwork(std::vector<int>& input_sizes, const std::vector<int>& output_sizes, float lr, int batch);
    ~NeuralNetwork();

    void forwardAndBackwardPass(float *input, float *target);
    void computeGradients();
    void updateWeights();
    void resetGradients();
    void computeLoss(float* target);
    std::vector<float> getWeights(int layerIndex);
    std::vector<float> getOutputs(int layerIndex);  
    
    void forwardPass(float *input);

    void computeGradientForLayer(int layerIndex);
    void applyGradientClipping(int layerIndex, float threshold);
    void updateWeightsForLayer(int layerIndex);
    void resetGradientsForLayer(int layerIndex);

};

#endif // NEURALNETWORK_H

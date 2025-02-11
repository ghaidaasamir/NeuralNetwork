#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> &input_sizes, const std::vector<int> &output_sizes, float lr, int batch)
    : layer_input_sizes(input_sizes), layer_output_sizes(output_sizes), learningRate(lr), batchSize(batch)
{
    unsigned long long seed = time(NULL);

    for (size_t i = 0; i < input_sizes.size(); i++)
    {
        int inSize = input_sizes[i];
        int outSize = output_sizes[i];

        float *weights;
        float *biases;
        float *weight_grad;
        float *bias_grad;
        float *output_grad;
        float *input_grad;
        float *layer_input;
        float *layer_output;

        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);

        cudaMalloc(&weights, inSize * outSize * sizeof(float));
        cudaMalloc(&biases, outSize * sizeof(float));
        cudaMalloc(&weight_grad, inSize * outSize * sizeof(float));
        cudaMalloc(&bias_grad, outSize * sizeof(float));
        cudaMalloc(&output_grad, batchSize * outSize * sizeof(float));
        cudaMalloc(&input_grad, batchSize * inSize * sizeof(float));
        cudaMalloc(&layer_input, batchSize * inSize * sizeof(float));
        cudaMalloc(&layer_output, batchSize * outSize * sizeof(float));

        std::vector<float> host_weights(inSize * outSize);
        std::default_random_engine generator;
        std::normal_distribution<float> distribution(0.0f, sqrt(2.0f / inSize));

        for (int j = 0; j < inSize * outSize; j++)
        {
            host_weights[j] = distribution(generator);
        }

        cudaMemcpy(weights, host_weights.data(), inSize * outSize * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(biases, 0, outSize * sizeof(float));

        d_weights.push_back(weights);
        d_biases.push_back(biases);
        d_weight_grad.push_back(weight_grad);
        d_bias_grad.push_back(bias_grad);
        d_output_grad.push_back(output_grad);
        d_input_grad.push_back(input_grad);
        d_layer_inputs.push_back(layer_input);
        d_layer_outputs.push_back(layer_output);
    }

    int inputSize = layer_input_sizes.front();
    int outputSize = layer_output_sizes.back();

    cudaMalloc(&d_input, batchSize * inputSize * sizeof(float));
    cudaMalloc(&d_output, batchSize * outputSize * sizeof(float));
    cudaMalloc(&d_target, batchSize * outputSize * sizeof(float));
}

NeuralNetwork::~NeuralNetwork()
{

    for (auto ptr : d_weights)
        cudaFree(ptr);
    for (auto ptr : d_biases)
        cudaFree(ptr);
    for (auto ptr : d_weight_grad)
        cudaFree(ptr);
    for (auto ptr : d_bias_grad)
        cudaFree(ptr);
    for (auto ptr : d_output_grad)
        cudaFree(ptr);
    for (auto ptr : d_input_grad)
        cudaFree(ptr);
    for (auto ptr : d_layer_inputs)
        cudaFree(ptr);
    for (auto ptr : d_layer_outputs)
        cudaFree(ptr);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_target);
}

void NeuralNetwork::forwardAndBackwardPass(float *input, float *target)
{
    forwardPass(input);

    computeLoss(target);

    // Compute gradients for all parameters
    computeGradients();

    // Update weights and biases
    updateWeights();

    // Reset gradients for next batch
    resetGradients();

    // Synchronize to ensure all operations are completed
    cudaDeviceSynchronize();
}

__global__ void computeFilterGradientsKernel(float *input, float *output_grad, float *filter_grad, int inputWidth, int inputHeight, int filterWidth, int filterHeight, int outputWidth, int outputHeight)
{
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        if (idx < filterWidth && idy < filterHeight)
        {
            float gradient_sum = 0;
            for (int oy = 0; oy < outputHeight; oy++)
            {
                for (int ox = 0; ox < outputWidth; ox++)
                {
                    int ix = ox + idx;
                    int iy = oy + idy;
                    if (ix < inputWidth && iy < inputHeight)
                    {
                        gradient_sum += input[iy * inputWidth + ix] * output_grad[oy * outputWidth + ox];
                    }
                }
            }
            filter_grad[idy * filterWidth + idx] = gradient_sum;
        }
    }
}

__global__ void computeWeightGradientsKernel(float *d_input, float *d_output_grad, float *d_weight_grad, int inputSize, int outputSize, int batchSize)
{
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    if (outX < inputSize && outY < outputSize)
    {
        float gradient = 0;
        for (int batch = 0; batch < batchSize; ++batch)
        {
            gradient += d_output_grad[batch * outputSize + outY] * d_input[batch * inputSize + outX];
        }
        atomicAdd(&d_weight_grad[outY * inputSize + outX], gradient);
    }
}

__global__ void computeBiasGradientsKernel(float *output_grad, float *bias_grad, int outputSize, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize)
    {
        float gradSum = 0.0f;
        for (int batch = 0; batch < batchSize; ++batch)
        {
            gradSum += output_grad[batch * outputSize + idx];
        }
        atomicAdd(&bias_grad[idx], gradSum);
    }
}

__global__ void clipGradientsKernel(float *gradients, int totalElements, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements)
    {
        gradients[idx] = fminf(fmaxf(gradients[idx], -threshold), threshold);
    }
}

__global__ void computeInputGradientsKernel(float *weights, float *output_gradient, float *input_gradient,
                                            int inputSize, int outputSize, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * inputSize;
    if (idx < totalElements)
    {
        int batch = idx / inputSize;
        int inputIdx = idx % inputSize;
        float grad = 0.0f;
        for (int out = 0; out < outputSize; ++out)
        {
            grad += weights[out * inputSize + inputIdx] * output_gradient[batch * outputSize + out];
        }
        input_gradient[batch * inputSize + inputIdx] = grad;
    }
}

__global__ void computeLossKernel(float *output, float *target, float *loss, int totalOutputs)
{
    // MSE
    // L=0.5∑n(yi-y^i)2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < totalOutputs)
    {
        float diff = output[idx] - target[idx];
        loss[idx] = 0.5f * diff * diff; // 0.5 * (predicted - actual)^2
    }
}

void NeuralNetwork::computeGradients()
{
    for (int i = layer_output_sizes.size() - 1; i >= 0; i--)
    {
        computeGradientForLayer(i);
    }
}

void NeuralNetwork::updateWeights()
{
    for (int i = 0; i < (int)layer_input_sizes.size(); i++)
    { // [FIX]
        updateWeightsForLayer(i);
    }
}

void NeuralNetwork::resetGradients()
{
    for (int i = 0; i < (int)layer_input_sizes.size(); i++)
    { // [FIX]
        resetGradientsForLayer(i);
    }
}

void NeuralNetwork::applyGradientClipping(int layerIndex, float threshold)
{
    float *d_layer_output_grad = d_output_grad[layerIndex];
    int totalElements = layer_output_sizes[layerIndex] * batchSize;

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    clipGradientsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_layer_output_grad, totalElements, threshold);
    cudaDeviceSynchronize();
}

void NeuralNetwork::computeGradientForLayer(int layerIndex)
{
    int inputSize = layer_input_sizes[layerIndex];
    int outputSize = layer_output_sizes[layerIndex];
    int nextOutputSize = (layerIndex + 1 < layer_output_sizes.size()) ? layer_output_sizes[layerIndex + 1] : 0;

    float *d_layer_input = d_layer_inputs[layerIndex];
    float *d_layer_output = d_layer_outputs[layerIndex];
    float *d_layer_output_grad = d_output_grad[layerIndex];
    float *d_layer_weight_grad = d_weight_grad[layerIndex];
    float *d_layer_bias_grad = d_bias_grad[layerIndex];

    cudaMemset(d_layer_weight_grad, 0, inputSize * outputSize * sizeof(float));
    cudaMemset(d_layer_bias_grad, 0, outputSize * sizeof(float));
    cudaMemset(d_layer_output_grad, 0, outputSize * batchSize * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;

    if (layerIndex < layer_output_sizes.size() - 1)
    {
        assert(layer_output_sizes[layerIndex] == layer_input_sizes[layerIndex + 1]);
    }

    if (layerIndex == layer_output_sizes.size() - 1)
    {
        computeGradientsKernel_last<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_output, d_target, d_layer_output_grad, outputSize, outputSize, batchSize);
    }
    else
    {
        computeGradientsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_output,
            d_output_grad[layerIndex + 1],
            d_layer_output_grad,
            d_weights[layerIndex + 1], // Next layer's weights
            outputSize,
            nextOutputSize,
            batchSize);
    }

    cudaDeviceSynchronize();

    float threshold = 1.0f;
    applyGradientClipping(layerIndex, threshold);

    dim3 weightBiasThreadsPerBlock(16, 16);
    dim3 weightBiasBlocksPerGrid(
        (inputSize + weightBiasThreadsPerBlock.x - 1) / weightBiasThreadsPerBlock.x,
        (outputSize + weightBiasThreadsPerBlock.y - 1) / weightBiasThreadsPerBlock.y);

    computeWeightGradientsKernel<<<weightBiasBlocksPerGrid, weightBiasThreadsPerBlock>>>(
        d_layer_input, d_layer_output_grad, d_layer_weight_grad, inputSize, outputSize, batchSize);
    cudaDeviceSynchronize();

    computeBiasGradientsKernel<<<(outputSize + 255) / 256, 256>>>(
        d_layer_output_grad, d_layer_bias_grad, outputSize, batchSize);
    cudaDeviceSynchronize();

    if (layerIndex > 0)
    {
        int currentInputSize = layer_input_sizes[layerIndex];
        int totalElements = batchSize * currentInputSize;
        int threadsPerBlockInput = 256;
        int blocksPerGridInput = (totalElements + threadsPerBlockInput - 1) / threadsPerBlockInput;

        computeInputGradientsKernel<<<blocksPerGridInput, threadsPerBlockInput>>>(
            d_weights[layerIndex], d_layer_output_grad, d_output_grad[layerIndex - 1],
            currentInputSize, outputSize, batchSize);
        cudaDeviceSynchronize();
    }
}

void NeuralNetwork::computeLoss(float *target)
{
    std::cout << "Target Array: ";
    for (int i = 0; i < 1; i++)
    {
        std::cout << target[i] << " ";
    }
    std::cout << std::endl;

    float *last_layer_output = d_layer_outputs.back();

    int threadsPerBlock = 256;
    int lastLayerIndex = layer_output_sizes.size() - 1;
    int outputSizeLastLayer = layer_output_sizes[lastLayerIndex];
    int totalOutputs = batchSize * outputSizeLastLayer;
    int blocksPerGrid = (totalOutputs + threadsPerBlock - 1) / threadsPerBlock;

    float *d_loss;
    cudaMalloc(&d_loss, totalOutputs * sizeof(float));
    cudaMemcpy(d_target, target, totalOutputs * sizeof(float), cudaMemcpyHostToDevice);

    computeLossKernel<<<blocksPerGrid, threadsPerBlock>>>(last_layer_output, d_target, d_loss, totalOutputs);
    cudaDeviceSynchronize();

    std::vector<float> h_loss(totalOutputs);
    cudaMemcpy(h_loss.data(), d_loss, totalOutputs * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = std::accumulate(h_loss.begin(), h_loss.end(), 0.0f) / totalOutputs;
    std::cout << "Total Loss: " << total_loss << std::endl;

    cudaFree(d_loss);
}

__global__ void updateWeightsKernel(float *weights, float *weight_grad, float learningRate, int weightSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < weightSize)
    {
        weights[index] = weights[index] - learningRate * weight_grad[index];
    }
}

__global__ void updateBiasesKernel(float *biases, float *bias_grad, float learningRate, int biasSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < biasSize)
    {
        biases[index] = biases[index] - learningRate * bias_grad[index];
    }
}

void NeuralNetwork::updateWeightsForLayer(int layerIndex)
{
    int inputSize = layer_input_sizes[layerIndex];
    int outputSize = layer_output_sizes[layerIndex];

    float *d_weights_layer = d_weights[layerIndex];
    float *d_biases_layer = d_biases[layerIndex];
    float *d_layer_weight_grad = d_weight_grad[layerIndex];

    float *d_layer_bias_grad = d_bias_grad[layerIndex];

    int threadsPerBlock = 256;
    int blocksPerGridWeights = (inputSize * outputSize + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridBiases = (outputSize + threadsPerBlock - 1) / threadsPerBlock;

    updateWeightsKernel<<<blocksPerGridWeights, threadsPerBlock>>>(d_weights_layer, d_layer_weight_grad, learningRate, inputSize * outputSize);
    updateBiasesKernel<<<blocksPerGridBiases, threadsPerBlock>>>(d_biases_layer, d_layer_bias_grad, learningRate, outputSize);
    cudaDeviceSynchronize();
}

void NeuralNetwork::resetGradientsForLayer(int layerIndex)
{
    int layerWeightSize = layer_input_sizes[layerIndex] * layer_output_sizes[layerIndex];
    int layerBiasSize = layer_output_sizes[layerIndex];

    float *d_layer_weight_grad = d_weight_grad[layerIndex];
    float *d_layer_bias_grad = d_bias_grad[layerIndex];

    cudaMemset(d_layer_weight_grad, 0, layerWeightSize * sizeof(float));
    cudaMemset(d_layer_bias_grad, 0, layerBiasSize * sizeof(float));
}

std::vector<float> NeuralNetwork::getOutputs(int layerIndex)
{

    int outputSize = layer_output_sizes[layerIndex];
    std::vector<float> host_output(outputSize * batchSize);

    cudaMemcpy(host_output.data(),
               d_layer_outputs[layerIndex],
               outputSize * batchSize * sizeof(float),
               cudaMemcpyDeviceToHost);

    return host_output;
}

std::vector<float> NeuralNetwork::getWeights(int layerIndex)
{

    int numWeights = layer_input_sizes[layerIndex] * layer_output_sizes[layerIndex];
    std::vector<float> host_weights(numWeights);
    cudaMemcpy(host_weights.data(), d_weights[layerIndex], numWeights * sizeof(float), cudaMemcpyDeviceToHost);

    return host_weights;
}

__global__ void forwardPassKernel(float *d_input, float *d_weights, float *d_biases, float *d_output,
                                  int inputSize, int outputSize, int batchSize, bool last_layer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize)
    {
        for (int b = 0; b < batchSize; b++)
        {
            float sum = 0.0f;
            for (int i = 0; i < inputSize; i++)
            {
                sum += d_input[b * inputSize + i] * d_weights[idx * inputSize + i];
            }

            sum += d_biases[idx];
            if (last_layer)
            {
                d_output[b * outputSize + idx] = sum;
            }

            else
                d_output[b * outputSize + idx] = max(0.1f, sum);
        }
    }
}

void NeuralNetwork::forwardPass(float *input)
{
    cudaMemcpy(d_layer_inputs[0], input, layer_input_sizes[0] * batchSize * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < d_weights.size(); i++)
    {
        int threadsPerBlock = 16;
        int blocksPerGrid = (layer_output_sizes[i] + threadsPerBlock - 1) / threadsPerBlock;
        bool last_layer = false;
        if (i == d_weights.size() - 1)
        {
            last_layer = true;
        }
        forwardPassKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_layer_inputs[i], d_weights[i], d_biases[i], d_layer_outputs[i], layer_input_sizes[i], layer_output_sizes[i], batchSize, last_layer);

        cudaDeviceSynchronize();

        if (i < d_weights.size() - 1)
        {
            cudaMemcpy(d_layer_inputs[i + 1], d_layer_outputs[i], layer_output_sizes[i] * batchSize * sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }

    cudaDeviceSynchronize();
}

__global__ void computeGradientsKernel_last(
    float *outputs,
    float *targets,
    float *grads,
    int outputSize,
    int batchSize,
    bool isOutputLayer)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outputSize)
    {
        for (int batch = 0; batch < batchSize; ++batch)
        {
            float output = outputs[batch * outputSize + idx];
            float target = targets[batch * outputSize + idx];
            grads[batch * outputSize + idx] = output - target; // MSE gradient
        }
    }
}

__global__ void computeGradientsKernel(
    float *current_outputs,
    float *next_layer_grads,
    float *current_grads,
    float *next_layer_weights,
    int current_size,
    int next_size,
    int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < current_size)
    {
        for (int batch = 0; batch < batchSize; ++batch)
        {
            float grad = 0.0f;

            for (int next_idx = 0; next_idx < next_size; ++next_idx)
            {
                int weight_index = idx * next_size + next_idx;
                int grad_index = batch * next_size + next_idx;
                grad += next_layer_grads[grad_index] * next_layer_weights[weight_index];
            }

            float output_val = current_outputs[batch * current_size + idx];
            grad *= (output_val > 0.0f) ? 1.0f : 0.1f;

            current_grads[batch * current_size + idx] = grad;
        }
    }
}
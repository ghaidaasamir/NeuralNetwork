#include "read_and_process_csv.h"
#include "neuralnetwork.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    std::vector<int> input_sizes  = {5}; 
    std::vector<int> output_sizes = {1};
    // std::vector<int> input_sizes  = {5, 3, 2};
    // std::vector<int> output_sizes = {3, 2, 1};
    float learningRate = 0.01f;
    int batchSize = 512;

    NeuralNetwork neuralNet(input_sizes, output_sizes, learningRate, batchSize);

    std::vector<std::vector<float>> inputBatches;
    std::vector<std::vector<float>> targetBatches;

    if (!loadCSVData("Student_Performance.csv", inputBatches, targetBatches, batchSize)) {
        std::cerr << "Failed to load CSV data." << std::endl;
        return 1;
    }

    // Train the network
    int epochs = 100;
    int decayFactor = 0.9;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "=== Epoch " << (epoch) << " ===" << std::endl;
        for (size_t batch = 0; batch < inputBatches.size(); ++batch) {
            std::cout << "=== Batch " << (batch) << " ===" << std::endl;
            neuralNet.forwardAndBackwardPass(inputBatches[batch].data(), targetBatches[batch].data());
            auto finalOutputs = neuralNet.getOutputs(output_sizes.size() - 1); // Last layer
            for (int i = 0; i < static_cast<int>(finalOutputs.size()); i++) {
                std::cout << "  output[" << i << "] = " << finalOutputs[i] << std::endl;
            }
        }
        learningRate *= decayFactor;
    }
   

    return 0;
}


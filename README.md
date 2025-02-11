# Neural Network Project

## Overview

This project implements a basic neural network in C++ using CUDA for parallel computation.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit (version 10.0 or later recommended)
- CMake 3.8 or higher

## Building the Project

1. Clone the repository.
2. Navigate to the project directory.
3. Create a build directory:
   ```bash
   mkdir build && cd build
   ```
4. Configure the project with CMake:
   ```bash
   cmake ..
   ```
5. Build the project:
   ```bash
   make
   ```

## Running the Project

2. Run the executable:
   ```bash
   ./Neuralnetwork
   ```

## Functionality

The program trains a neural network using data from `Student_Performance.csv`. It performs the following operations:
- Loads and processes batched input data.
- Conducts forward and backward passes through the network.
- Updates weights and biases based on computed gradients.
- Outputs the network predictions for each batch and epoch.

## Customizing the Network

To adjust the neural network's architecture:
- Edit the `input_sizes` and `output_sizes` in the `main()` function to specify the size of each layer.

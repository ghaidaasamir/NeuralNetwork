#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>

float yesNoToFloat(const std::string& value) {
    if (value == "Yes") return 1.0f;
    else if (value == "No") return 0.0f;
}


bool loadCSVData(const std::string& filename, 
                 std::vector<std::vector<float>>& inputBatches, 
                 std::vector<std::vector<float>>& targetBatches, 
                 int batchSize) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line); // Skip header
    std::vector<float> currentInputBatch;
    std::vector<float> currentTargetBatch;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        int columnIndex = 0;

        while (std::getline(ss, cell, ',')) {
            if (columnIndex < 5) {
                float value = (columnIndex == 2) ? yesNoToFloat(cell) : std::stof(cell);
                row.push_back(value); 
            } else if (columnIndex == 5) {
                float target = std::stof(cell);
                currentTargetBatch.push_back(target);
                break;
            }
            columnIndex++;
        }

        currentInputBatch.insert(currentInputBatch.end(), row.begin(), row.end());

        if (currentTargetBatch.size() == batchSize) {
            inputBatches.push_back(currentInputBatch);
            targetBatches.push_back(currentTargetBatch);
            currentInputBatch.clear();
            currentTargetBatch.clear();
        }
    }

    if (!currentTargetBatch.empty()) {
        inputBatches.push_back(currentInputBatch);
        targetBatches.push_back(currentTargetBatch);
    }

    return true;
}
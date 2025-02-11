#ifndef CSV_READER_H
#define CSV_READER_H

#include <vector>
#include <string>

// Function to load data from a CSV file, specified by filename
// It will read the first five columns into 'data' and the sixth column into 'targets'
bool loadCSVData(const std::string& filename, 
                 std::vector<std::vector<float>>& inputBatches, 
                 std::vector<std::vector<float>>& targetBatches, 
                 int batchSize);

#endif // CSV_READER_H

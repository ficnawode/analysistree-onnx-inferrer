#pragma once
#include <vector>
#include "ONNXRunner.hpp"

class ONNXSingleBinModel
{
public:
    ONNXSingleBinModel(float min, float max, std::string model_path);
    ~ONNXSingleBinModel();

    float GetMin();
    float GetMax();

    std::vector<float> PredictSingleInstance(std::vector<float> &feature_values);
    std::vector<std::vector<float>> PredictMany(std::vector<std::vector<float>> &feature_values);

private:
    float min;
    float max;
    ONNXRunner *onnx_runner;
};
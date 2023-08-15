#pragma once
#include <vector>
#include <array>

#include "ONNXRunner.hpp"

class ONNXSingleBinModel
{
public:
    ONNXSingleBinModel(float min, float max, std::string model_file_path, size_t num_features, size_t num_threads = 1);
    ~ONNXSingleBinModel() = default;

    float GetMin();
    float GetMax();

    std::array<size_t, 2> GetOutputTensorShape();
    size_t GetOutputTensorSize();

    std::vector<float> InferSingle(std::vector<float> &feature_values);

private:
    void InitRunner();

    float min_;
    float max_;
    size_t num_threads_;
    size_t num_features_;
    std::string model_file_path_;
    std::array<size_t, 2> output_tensor_shape_;
    size_t output_tensor_size_;

    ONNXRunner *onnx_runner_;
};
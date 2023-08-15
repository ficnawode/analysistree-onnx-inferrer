#pragma once
#include "ONNXConfigParser.hpp"
#include "ONNXSingleBinModel.hpp"
#include <vector>

class ONNXConfigManager
{
public:
    ONNXConfigManager(std::string json_config_path);
    ~ONNXConfigManager() = default;

    std::vector<std::vector<float>> InferMultiple(std::vector<std::vector<float>> features, std::vector<float> momentums);

    std::vector<std::string> GetFeatureFieldNames();
    std::array<size_t, 2> GetOutputTensorShape();
    size_t GetOutputTensorSize();

private:
    std::vector<float> InferSingle(std::vector<float> features, float momentuum);
    void LoadBinModels(std::vector<BinConfig>);

    std::vector<ONNXSingleBinModel> bin_models_;
    std::vector<std::string> feature_field_names_;
    std::array<size_t, 2> output_tensor_shape_;
    size_t output_tensor_size_;
};
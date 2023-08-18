#pragma once
#include "ONNXConfigParser.hpp"
#include "ONNXSingleBinModel.hpp"
#include <vector>

class ONNXConfigManager
{
public:
    ONNXConfigManager(std::string json_config_path);
    ~ONNXConfigManager() = default;

    std::vector<std::vector<float>> InferMultiple(std::vector<std::vector<float>> features_vector, std::vector<float> momentums);
    std::vector<float> InferSingle(std::vector<float> features, float momentum);

    std::vector<std::string> GetFeatureFieldNames() { return feature_field_names_; };
    std::array<size_t, 2> GetOutputTensorShape() { return output_tensor_shape_; }
    size_t GetOutputTensorSize() { return output_tensor_size_; }

private:
    void LoadBinModels(std::vector<BinConfig>);

    std::vector<ONNXSingleBinModel> bin_models_;
    std::vector<std::string> feature_field_names_;
    std::array<size_t, 2> output_tensor_shape_;
    size_t output_tensor_size_;
};
#include "ONNXConfigManager.hpp"
#include "ONNXConfigParser.hpp"

#include <iostream>

ONNXConfigManager::ONNXConfigManager(std::string json_config_path)
{
    auto parser = ONNXConfigParser(json_config_path);
    feature_field_names_ = parser.ParseModelFeatures();
    auto bin_configs = parser.ParseBinConfigs();

    LoadBinModels(bin_configs);
    assert(bin_models_.size() > 0);

    output_tensor_shape_ = bin_models_[0].GetOutputTensorShape();
    output_tensor_size_ = bin_models_[0].GetOutputTensorSize();
}

std::vector<float> ONNXConfigManager::InferSingle(std::vector<float> features, float momentum)
{
    for (auto &bin_model : bin_models_)
    {
        if (momentum > bin_model.GetMin() && momentum <= bin_model.GetMax())
        {
            return bin_model.InferSingle(features);
        }
    }
    return {};
}

std::vector<std::vector<float>> ONNXConfigManager::InferMultiple(std::vector<std::vector<float>> features, std::vector<float> momentums)
{
    assert(features.size() == momentums.size());
    std::vector<std::vector<float>>
        inferred_values;
    for (size_t i = 0; i < features.size(); i++)
    {
        inferred_values.push_back(InferSingle(features[i], momentums[i]));
    }
    return inferred_values;
}

std::vector<std::string> ONNXConfigManager::GetFeatureFieldNames() { return feature_field_names_; }
std::array<size_t, 2> ONNXConfigManager::GetOutputTensorShape() { return output_tensor_shape_; }
size_t ONNXConfigManager::GetOutputTensorSize() { return output_tensor_size_; }

void ONNXConfigManager::LoadBinModels(std::vector<BinConfig> bin_configs)
{
    size_t num_features = feature_field_names_.size();
    for (auto &config : bin_configs)
    {
        auto min = config.min;
        auto max = config.max;
        auto path = config.path;
        auto bin_model = ONNXSingleBinModel(min, max, path, num_features);
        bin_models_.push_back(bin_model);
    }
}
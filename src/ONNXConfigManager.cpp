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
    std::vector<float> zeros(output_tensor_size_);
    return zeros;
}

std::vector<std::vector<float>> ONNXConfigManager::InferMultiple(std::vector<std::vector<float>> features_vector, std::vector<float> momentums)
{
    assert(features_vector.size() == momentums.size());
    std::vector<std::vector<float>> inferred_tensors;
    for (size_t i = 0; i < features_vector.size(); i++)
    {
        inferred_tensors.push_back(InferSingle(features_vector[i], momentums[i]));
    }
    return inferred_tensors;
}

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
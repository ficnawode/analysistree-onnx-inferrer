#include "ONNXConfigManager.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <iostream>

ONNXConfigManager::ONNXConfigManager(std::string json_config_path) : json_config_path_{json_config_path}
{
    auto bin_configs = ParseBinConfigs(json_config_path_);
    LoadBinModels(bin_configs);
}

std::vector<float> ONNXConfigManager::InferSingle(std::vector<float> features, float momentum)
{
    for (auto &bin_model : bin_models_)
    {
        // if (momentum > bin_model.GetMin() && momentum <= bin_model.GetMax())
        // {
        // return bin_model.PredictSingleInstance(features);
        // }
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

std::vector<ONNXConfigManager::BinConfig> ONNXConfigManager::ParseBinConfigs(std::string json_config_path)
{
    namespace pt = boost::property_tree;
    std::vector<BinConfig> bin_configs;
    pt::ptree root;
    pt::read_json(json_config_path, root);
    for (pt::ptree::value_type &model_dict_val : root.get_child("model_paths"))
    {
        pt::ptree model_dict = model_dict_val.second;
        float min = model_dict.get<float>("lo");
        float max = model_dict.get<float>("hi");
        std::string path = model_dict.get<std::string>("path");

        BinConfig bin_config{min, max, path};
        bin_configs.push_back(bin_config);
    }
    std::cout << "Loaded ranges and paths from " << json_config_path << ':' << std::endl;
    for (auto &config : bin_configs)
    {
        std::cout << '(' << config.min << " : " << config.max << ']' << " -> " << config.path << std::endl;
    }
    return bin_configs;
}

void ONNXConfigManager::LoadBinModels(std::vector<ONNXConfigManager::BinConfig>) {}
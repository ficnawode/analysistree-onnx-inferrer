#include "ONNXConfigParser.hpp"
#include <iostream>

namespace pt = boost::property_tree;

ONNXConfigParser::ONNXConfigParser(std::string json_config_path) : json_config_path_{json_config_path}
{
    pt::read_json(json_config_path, root_);
}

std::vector<BinConfig> ONNXConfigParser::ParseBinConfigs()
{
    std::vector<BinConfig> bin_configs;
    for (pt::ptree::value_type &model_dict_val : root_.get_child("model_paths"))
    {
        pt::ptree model_dict = model_dict_val.second;
        float min = model_dict.get<float>("lo");
        float max = model_dict.get<float>("hi");
        std::string path = model_dict.get<std::string>("path");

        BinConfig bin_config{min, max, path};
        bin_configs.push_back(bin_config);
    }
    std::cout << "Loaded ranges and paths from " << json_config_path_ << ':' << std::endl;
    for (auto &config : bin_configs)
    {
        std::cout << '(' << config.min << " : " << config.max << ']' << " -> " << config.path << std::endl;
    }
    return bin_configs;
}

std::vector<std::string> ONNXConfigParser::ParseModelFeatures()
{
    std::vector<std::string> model_features;
    for (pt::ptree::value_type &feature_name : root_.get_child("model_features"))
    {
        std::string model_feature = feature_name.second.data();
        model_features.push_back(model_feature);
    }
    std::cout << "Loaded feature variables " << json_config_path_ << ':' << std::endl;
    for (auto &feature : model_features)
    {
        std::cout << feature << std::endl;
    }
    return model_features;
}
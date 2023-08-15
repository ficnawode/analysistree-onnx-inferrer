#pragma once
#include "ONNXSingleBinModel.hpp"
#include <vector>

class ONNXConfigManager
{
public:
    ONNXConfigManager(std::string json_config_path);
    ~ONNXConfigManager() = default;

    std::vector<float> InferSingle(std::vector<float> features, float momentuum);
    std::vector<std::vector<float>> InferMultiple(std::vector<std::vector<float>> features, std::vector<float> momentums);

private:
    struct BinConfig
    {
        float min;
        float max;
        std::string path;
    };
    static std::vector<BinConfig> ParseBinConfigs(std::string json_config_path);
    void LoadBinModels(std::vector<BinConfig>);

    std::string json_config_path_;
    std::vector<ONNXSingleBinModel> bin_models_;
};
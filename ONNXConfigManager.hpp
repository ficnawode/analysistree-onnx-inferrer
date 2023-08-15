#pragma once
#include <vector>
#include "ONNXRunner.hpp"

class ONNXConfigManager
{
public:
    ONNXConfigManager(std::string json_config_path);
    ~ONNXConfigManager() = default;

    std::vector<float> InferSingle(std::vector<float> features);
    std::vector<std::vector<float>> InferMultiple(std::vector<std::vector<float>> features);

private:
    struct BinConfig
    {
        float min;
        float max;
        std::string model_path;
    };
    std::vector<BinConfig> ParseBinConfigs(std::string json_config_path);
    void LoadBinModels(std::vector<BinConfig>);
};
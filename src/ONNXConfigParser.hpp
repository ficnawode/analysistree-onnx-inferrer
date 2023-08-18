#pragma once
#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

struct BinConfig
{
    float min;
    float max;
    std::string path;
};

class ONNXConfigParser
{
public:
    ONNXConfigParser(std::string json_config_path);

    std::vector<BinConfig> ParseBinConfigs();
    std::vector<std::string> ParseModelFeatures();

private:
    boost::property_tree::ptree root_;
    std::string json_config_path_;
};
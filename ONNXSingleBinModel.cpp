#include "ONNXSingleBinModel.hpp"
#include <iostream>

ONNXSingleBinModel::ONNXSingleBinModel(float min, float max, std::string model_file_path, size_t num_features, size_t num_threads) : min_{min}, max_{max}, num_threads_{num_threads}, num_features_{num_features}, model_file_path_{model_file_path}
{
    InitRunner();
    output_tensor_shape_ = onnx_runner_->GetOutputTensorShape();
    output_tensor_size_ = output_tensor_shape_[0] * output_tensor_shape_[1];
}

float ONNXSingleBinModel::GetMin() { return min_; }
float ONNXSingleBinModel::GetMax() { return max_; }

std::array<size_t, 2> ONNXSingleBinModel::GetOutputTensorShape() { return output_tensor_shape_; }
size_t ONNXSingleBinModel::GetOutputTensorSize() { return output_tensor_size_; }

void ONNXSingleBinModel::InitRunner()
{
    std::cout << "Loading ONNX model file " << model_file_path_ << std::endl;
    onnx_runner_ = new ONNXRunner();
    onnx_runner_->Init(model_file_path_, num_threads_);

    if (onnx_runner_->GetFeatureCount() != num_features_)
    {
        std::cout << "ERROR: ONNX Model requires " << onnx_runner_->GetFeatureCount() << " features, but " << num_features_ << " are given!" << std::endl;
        exit(-1);
    }
}

std::vector<float> ONNXSingleBinModel::InferSingle(std::vector<float> &feature_values)
{
    return onnx_runner_->PredictSingleInstance(feature_values);
}

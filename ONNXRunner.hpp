#ifndef ONNXRUNNER_HPP
#define ONNXRUNNER_HPP

#include "onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include <array>

class ONNXRunner
{
public:
  ONNXRunner();
  ~ONNXRunner() = default;

  void Init(std::string model_file, int num_threads = -1);
  std::vector<float> PredictSingleInstance(std::vector<float> &feature_values);
  std::vector<std::vector<float>> PredictMany(std::vector<std::vector<float>> &feature_values);

  int GetFeatureCount() { return feature_count_; }
  std::array<size_t, 2> GetOutputTensorShape() { return output_tensor_shape_; }
  size_t GetOutputTensorSize() { return output_tensor_size_; }

private:
  Ort::Env *env_ = nullptr;
  Ort::Experimental::Session *session_ = nullptr;
  int total_number_of_elements_per_run_ = 0;
  int feature_count_ = 0;

  std::array<size_t, 2> output_tensor_shape_;
  size_t output_tensor_size_;

  std::string print_shape(const std::vector<int64_t> &v);
  int calculate_product(const std::vector<int64_t> &v);
};
#endif // ONNXRUNNER_HPP
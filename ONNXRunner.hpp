#ifndef ONNXRUNNER_HPP
#define ONNXRUNNER_HPP

//#include "onnxruntime-linux-x64/include/onnxruntime/core/session/experimental_onnxruntime_cxx_api.h"
#include "include/onnx/core/session/experimental_onnxruntime_cxx_api.h"
/*
namespace Ort
{
  namespace Experimental
  {
    class Session;
  }
}
*/

class ONNXRunner {
 public:
  ONNXRunner();
  ~ONNXRunner() = default;
 
  void Init(std::string model_file, int num_threads = -1);
  float PredictSingleInstance(std::vector<float> feature_values);
  std::vector<float> PredictBatch(std::vector<float> feature_values);
  int GetFeatureCount() {return feature_count_;}
  
private:
  Ort::Env* env_ = nullptr;
  Ort::Experimental::Session* session_ = nullptr;
  int total_number_of_elements_per_run_ = 0;
  int feature_count_ = 0;
  
  std::string print_shape(const std::vector<int64_t>& v);
  int calculate_product(const std::vector<int64_t>& v);
  
  
};
#endif // ONNXRUNNER_HPP
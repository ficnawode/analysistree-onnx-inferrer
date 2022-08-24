#include "ONNXRunner.hpp"
#include <iostream>
#include <sstream>
#include <cassert>
#include <math.h>
//#include <experimental_onnxruntime_cxx_api.h>

ONNXRunner::ONNXRunner()
{
  
}

// pretty prints a shape dimension vector
std::string ONNXRunner::print_shape(const std::vector<int64_t>& v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int ONNXRunner::calculate_product(const std::vector<int64_t>& v)
{
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

void ONNXRunner::Init(std::string model_file, int num_threads)
{
  env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "atree-prediction-adder");
  Ort::SessionOptions session_options;
  if (num_threads > 0)
    session_options.SetIntraOpNumThreads(num_threads);
  session_ = new Ort::Experimental::Session(*env_, model_file, session_options);
  
  auto input_names = session_->GetInputNames();
  auto input_shapes = session_->GetInputShapes();
  
  std::cout << "ONNXRunner Init Info:" << std::endl;
  std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
  for (size_t i = 0; i < input_names.size(); i++) {
    std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
  }

  // print name/shape of outputs
  auto output_names = session_->GetOutputNames();
  auto output_shapes = session_->GetOutputShapes();
  std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
  for (size_t i = 0; i < output_names.size(); i++) {
    std::cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
  }
  
  auto input_shape = input_shapes[0];
  total_number_of_elements_per_run_ = calculate_product(input_shape);
  feature_count_ = input_shape[0];
}

float ONNXRunner::PredictSingleInstance(std::vector<float> feature_values)
{
  auto input_shapes = session_->GetInputShapes();
  auto input_shape = input_shapes[0];
  
  std::vector<float> input_tensor_values(total_number_of_elements_per_run_);
  
  for (int i = 0; i < total_number_of_elements_per_run_; ++i)
    if (i < feature_values.size())
      input_tensor_values[i] = feature_values[i];
    else
      input_tensor_values[i] = 0.0f;
  
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));

  // pass data through model
  try {
    auto output_tensors = session_->Run(session_->GetInputNames(), input_tensors, session_->GetOutputNames());

    float* output_tensor_values = output_tensors[1].GetTensorMutableData<float>();
    
    return output_tensor_values[1];

  } catch (const Ort::Exception& exception) {
    std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }
  
  return 0;
}

std::vector<float> ONNXRunner::PredictBatch(std::vector<float> feature_values)
{
  std::vector<float> signal_prob;
    
  auto input_shapes = session_->GetInputShapes();
  auto input_shape = input_shapes[0];
  
  int candidate_id = 0;
  int candidates_per_tensor = input_shape[1]; //each tensor contains multiple candidates
  int candidate_count = feature_values.size()/feature_count_;
  
  if (candidate_count == 0)
    return signal_prob;
  
  int tensor_count = ceil((1.0*candidate_count)/candidates_per_tensor);
  
  //printf("Predicting %d candidates with %d tensors\n", candidate_count, tensor_count);
  
  
  for (int iTensor = 0; iTensor < tensor_count; ++iTensor)
  {
    std::vector<Ort::Value> input_tensors;
  
    std::vector<float> input_tensor_values(total_number_of_elements_per_run_);
    
    for (int iCandidate = 0; iCandidate < candidates_per_tensor; ++iCandidate)
    {
      if (candidate_id >= candidate_count)
        break;
      
      for (int iFeature = 0; iFeature < feature_count_; ++iFeature)
        input_tensor_values[iCandidate*feature_count_ + iFeature] = feature_values[candidate_id*feature_count_ + iFeature];
      
      ++candidate_id;
    }
    
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
    
    // pass data through model
    try {
      auto output_tensors = session_->Run(session_->GetInputNames(), input_tensors, session_->GetOutputNames());

      float* output_tensor_values = output_tensors[1].GetTensorMutableData<float>();
      
      for (int iCandidate = 0; iCandidate < candidates_per_tensor; ++iCandidate)
        signal_prob.push_back(output_tensor_values[iCandidate*2 + 1]);

    } catch (const Ort::Exception& exception) {
      std::cout << "ERROR running model inference: " << exception.what() << std::endl;
      exit(-1);
    }
    
  }

  
  return signal_prob;
}
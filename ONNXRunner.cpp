#include "ONNXRunner.hpp"
#include <iostream>
#include <sstream>
#include <cassert>
#include <math.h>
#include <cstring>

ONNXRunner::ONNXRunner()
{
}

std::string ONNXRunner::print_shape(const std::vector<int64_t> &v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int ONNXRunner::calculate_product(const std::vector<int64_t> &v)
{
  int total = 1;
  for (auto &i : v)
    total *= i;
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
  for (size_t i = 0; i < input_names.size(); i++)
  {
    std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
  }

  // print name/shape of outputs
  auto output_names = session_->GetOutputNames();
  auto output_shapes = session_->GetOutputShapes();

  output_tensor_shape_ = {output_shapes[1][0], output_shapes[1][1]};
  output_tensor_size_ = output_tensor_shape_[0] * output_tensor_shape_[1];
  std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
  for (size_t i = 0; i < output_names.size(); i++)
  {
    std::cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
  }

  auto input_shape = input_shapes[0];
  total_number_of_elements_per_run_ = calculate_product(input_shape);
  feature_count_ = input_shape[0];
}

std::vector<float> ONNXRunner::PredictSingleInstance(std::vector<float> &feature_values)
{
  assert(feature_values.size() > 0);
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
  try
  {
    auto output_tensors = session_->Run(session_->GetInputNames(), input_tensors, session_->GetOutputNames());
    float *tensor_data = output_tensors[1].GetTensorMutableData<float>();
    std::vector<float> tensor(output_tensor_size_);
    std::memcpy(tensor.data(), tensor_data, output_tensor_size_);
    return tensor;
  }
  catch (const Ort::Exception &exception)
  {
    std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    exit(-1);
  }

  return {};
}

std::vector<std::vector<float>> ONNXRunner::PredictMany(std::vector<std::vector<float>> &feature_values_vector)
{
  std::vector<std::vector<float>> tensor_vector;
  for (auto &feature_values : feature_values_vector)
  {
    auto tensor = PredictSingleInstance(feature_values);
    assert(tensor.size() > 0);
    tensor_vector.push_back(tensor);
  }
  return tensor_vector;
}

#ifndef ATREEPREDICTIONADDER_HPP
#define ATREEPREDICTIONADDER_HPP

#include "AnalysisTree/Cuts.hpp"
#include "AnalysisTree/Detector.hpp"
#include "AnalysisTree/Task.hpp"
#include "ONNXRunner.hpp"
#include "ONNXConfigManager.hpp"
#include <array>

class ATreePredictionAdder : public AnalysisTree::Task
{
public:
  explicit ATreePredictionAdder() = default;
  ~ATreePredictionAdder() override = default;

  void Init() override;
  void Exec() override;
  void Finish() override{};

  void InitIndices();

  void SetCuts(AnalysisTree::Cuts *cuts) { cuts_ = cuts; }
  void SetInputBranchName(std::string input_branch_name) { input_branch_name_ = input_branch_name; }
  void SetOutputBranchName(std::string output_branch_name) { output_branch_name_ = output_branch_name; }
  void SetFeatureFieldNames(std::string feature_field_name_arg) { feature_field_names_ = stringSplit(feature_field_name_arg, ","); }
  void SetModelFileName(std::string model_file_name) { model_file_name_ = model_file_name; }
  void SetNumThreads(int num_threads) { num_threads_ = num_threads; }
  void SetONNXConfigPath(std::string onnx_config_path) { onnx_config_path_ = onnx_config_path; }

private:
  ONNXConfigManager *onnx_model_manager_{nullptr};
  std::string onnx_config_path_;

  // input branch
  AnalysisTree::Particles *in_branch_{nullptr};

  // output branch
  AnalysisTree::Particles *out_branch_{nullptr};

  AnalysisTree::Cuts *cuts_{nullptr};

  std::string input_branch_name_{"Candidates_plain"};
  std::string output_branch_name_{"Candidates_plainPredicted"};
  std::vector<std::string> feature_field_names_;
  std::vector<int> feature_field_ids_;
  std::string model_file_name_{"model_onnx.onnx"};
  int num_threads_ = -1;

  //**** input fields ***********
  int mass2_first_field_id_r_{AnalysisTree::UndefValueInt};
  int mass2_second_field_id_r_{AnalysisTree::UndefValueInt};
  int momentum_id_{AnalysisTree::UndefValueInt};
  //*****************************

  //***** output fields *********

  std::vector<std::string> stringSplit(std::string s, std::string delimiter);
  std::array<size_t, 2> output_tensor_shape_;
  size_t output_tensor_buffer_size_;
  std::vector<std::string> output_field_names_;
  std::vector<int> output_field_ids_;

  void InitFeatureIds();
  void InitModel();

  void FillOutputTensorShape();
  void FillOutputTensorSize();
  void FillOutputFieldNames();
  void SetTensorFields(AnalysisTree::Particle &particle, std::vector<float> tensor);
  std::vector<std::vector<float>> ExecGetONNXFeatureValues();
};
#endif // ATREEPREDICTIONADDER_HPP
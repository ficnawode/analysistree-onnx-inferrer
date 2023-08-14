#ifndef ATREEPREDICTIONADDER_HPP
#define ATREEPREDICTIONADDER_HPP

#include "AnalysisTree/Cuts.hpp"
#include "AnalysisTree/Detector.hpp"
#include "AnalysisTree/Task.hpp"
#include "ONNXRunner.hpp"
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

protected:
  ONNXRunner *onnx_runner_;

  // input branches
  AnalysisTree::Particles *candidates_{nullptr};

  // output branch
  AnalysisTree::Particles *plain_branch_{nullptr};

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
  // int generation_field_id_r_{AnalysisTree::UndefValueInt};
  //*****************************

  //***** output fields *********
  int onnx_pred_field_id_w_{AnalysisTree::UndefValueInt};

  std::vector<std::string> stringSplit(std::string s, std::string delimiter);

private:
  std::array<size_t, 2> output_tensor_shape_;
  size_t output_tensor_buffer_size_;
  std::vector<std::string> tensor_field_names_;

  void InitFeatureIds();
  void InitModel();

  void FillOutputTensorShape();
  void FillOutputTensorSize();
  std::vector<std::vector<float>> ExecGetONNXFeatureValues();
};
#endif // ATREEPREDICTIONADDER_HPP
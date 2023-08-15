#include "ATreePredictionAdder.hpp"
#include "AnalysisTree/TaskManager.hpp"
#include "boost/json.hpp"

void ATreePredictionAdder::Init()
{
  std::cout << "ATreePredictionAdder Init:" << std::endl;
  auto *man = AnalysisTree::TaskManager::GetInstance();
  auto *chain = man->GetChain();

  candidates_ = ANALYSISTREE_UTILS_GET<AnalysisTree::Particles *>(chain->GetPointerToBranch(input_branch_name_));

  auto out_config = AnalysisTree::TaskManager::GetInstance()->GetConfig();

  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);
  std::string momentum_name = "p";
  momentum_id_ = in_branch_cand.GetFieldId(momentum_name);
  AnalysisTree::BranchConfig out_particles = in_branch_cand.Clone(output_branch_name_, in_branch_cand.GetType());

  InitFeatureIds();
  InitModel();
  FillOutputTensorShape();
  FillOutputTensorSize();
  FillOutputFieldNames();

  for (auto name : output_field_names_)
  {
    out_particles.AddField<float>(name.c_str(), "");
    std::cout << "Field added: " << name << std::endl;
  }

  std::cout << "Input Branch Config:" << std::endl;
  in_branch_cand.Print();

  man->AddBranch(plain_branch_, out_particles);

  std::cout << "Output Branch Config:" << std::endl;
  out_particles.Print();

  if (cuts_)
    cuts_->Init(*out_config);

  InitIndices();
}

void ATreePredictionAdder::FillOutputTensorShape()
{
  output_tensor_shape_ = onnx_model_manager_->GetOutputTensorShape();
}

void ATreePredictionAdder::FillOutputTensorSize()
{
  output_tensor_buffer_size_ = output_tensor_shape_[0] * output_tensor_shape_[1];
}

void ATreePredictionAdder::FillOutputFieldNames()
{
  for (int i = 0; i < output_tensor_shape_[0]; i++)
  {
    for (int j = 0; j < output_tensor_shape_[1]; j++)
    {

      std::string field_name = "onnx_pred_" + std::to_string(i) + '_' + std::to_string(j);
      output_field_names_.push_back(field_name);
    }
  }
}

void ATreePredictionAdder::InitFeatureIds()
{
  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);
  for (auto &feature_field_name : feature_field_names_)
    feature_field_ids_.push_back(in_branch_cand.GetFieldId(feature_field_name));
}

void ATreePredictionAdder::InitModel()
{
  onnx_model_manager_ = new ONNXConfigManager(onnx_config_path_);
}

void ATreePredictionAdder::Exec()
{
  plain_branch_->ClearChannels();

  auto out_config = AnalysisTree::TaskManager::GetInstance()->GetConfig();

  // Predict probabilities with ONNX model and save to signal_probs vector

  auto *man = AnalysisTree::TaskManager::GetInstance();
  auto *chain = man->GetChain();
  std::vector<std::vector<float>> onnxFeatureValues;
  std::vector<float> onnxMomentumValues;

  for (auto &input_particle : *candidates_)
  {
    std::vector<float> particle_feature_values;
    for (auto &feature_field_id : feature_field_ids_)
    {
      float feature_value = input_particle.GetField<float>(feature_field_id);
      if (feature_value == -999.0f && (feature_field_id == mass2_first_field_id_r_ || feature_field_id == mass2_second_field_id_r_))
        feature_value = nan("");

      particle_feature_values.push_back(feature_value);
    }
    float momentum_value = input_particle.GetField<float>(momentum_id_);
    onnxFeatureValues.push_back(particle_feature_values);
    onnxMomentumValues.push_back(momentum_value);
  }

  auto outputTensors = onnx_model_manager_->InferMultiple(onnxFeatureValues, onnxMomentumValues);

  // Add predictions to output candidates
  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);
  int iCandidate = -1;

  for (auto &input_particle : *candidates_)
  {
    ++iCandidate;
    if (cuts_)
      if (!cuts_->Apply(input_particle))
        continue;

    auto &output_particle = plain_branch_->AddChannel(out_config->GetBranchConfig(plain_branch_->GetId()));

    // Add ONNX prediction
    auto outputTensor = outputTensors[iCandidate];
    SetTensorFields(output_particle, outputTensor);
    // output_particle.SetField(outputTensor[2], onnx_pred_field_id_w_);

    // Copy all other fields for the candidate
    for (const auto &field : in_branch_cand.GetMap<float>())
      output_particle.SetField(input_particle.GetField<float>(field.second.id_), field.second.id_);

    for (const auto &field : in_branch_cand.GetMap<int>())
      output_particle.SetField(input_particle.GetField<int>(field.second.id_), field.second.id_);

    for (const auto &field : in_branch_cand.GetMap<bool>())
      output_particle.SetField(input_particle.GetField<bool>(field.second.id_), field.second.id_);
  }
}

void ATreePredictionAdder::SetTensorFields(AnalysisTree::Particle particle, std::vector<float> tensor)
{
  if (tensor.size() != output_tensor_buffer_size_)
  {
    std::string error_message = std::to_string(tensor.size()) + "!=" + std::to_string(output_tensor_buffer_size_);
    throw std::runtime_error(error_message);
  }
  for (size_t i = 0; i < output_tensor_buffer_size_; i++)
  {
    particle.SetField(tensor[i], output_field_ids_[i]);
  }
}

std::vector<std::vector<float>> ATreePredictionAdder::ExecGetONNXFeatureValues()
{
  std::vector<std::vector<float>> onnxFeatureValues;

  for (auto &input_particle : *candidates_)
  {
    std::vector<float> particle_feature_values;
    for (auto &feature_field_id : feature_field_ids_)
    {
      float feature_value = input_particle.GetField<float>(feature_field_id);
      if (feature_value == -999.0f && (feature_field_id == mass2_first_field_id_r_ || feature_field_id == mass2_second_field_id_r_))
        feature_value = nan("");

      particle_feature_values.push_back(feature_value);
    }
    onnxFeatureValues.push_back(particle_feature_values);
  }
  assert(onnxFeatureValues.size() > 0);
  return onnxFeatureValues;
}

void ATreePredictionAdder::InitIndices()
{
  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);

  mass2_first_field_id_r_ = in_branch_cand.GetFieldId("mass2_first");
  mass2_second_field_id_r_ = in_branch_cand.GetFieldId("mass2_second");
  // generation_field_id_r_       = in_branch_cand.GetFieldId("generation");

  auto out_config = AnalysisTree::TaskManager::GetInstance()->GetConfig();
  const auto &out_branch = out_config->GetBranchConfig(plain_branch_->GetId());

  for (auto name : output_field_names_)
  {
    int temp_id = out_branch.GetFieldId(name);
    output_field_ids_.push_back(temp_id);
  }
}

std::vector<std::string> ATreePredictionAdder::stringSplit(std::string s, std::string delimiter)
{
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
  {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

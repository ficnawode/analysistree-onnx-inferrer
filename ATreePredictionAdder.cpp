#include "ATreePredictionAdder.hpp"
#include "AnalysisTree/TaskManager.hpp"

void ATreePredictionAdder::Init()
{
  std::cout << "ATreePredictionAdder Init:" << std::endl;
  auto* man = AnalysisTree::TaskManager::GetInstance();
  auto* chain = man->GetChain();
  
  candidates_ = ANALYSISTREE_UTILS_GET<AnalysisTree::Particles*>(chain->GetPointerToBranch(input_branch_name_));
  
  auto out_config = AnalysisTree::TaskManager::GetInstance()->GetConfig();
  
  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);
  AnalysisTree::BranchConfig out_particles = in_branch_cand.Clone(output_branch_name_, in_branch_cand.GetType());
  out_particles.AddField<float>("onnx_pred", "");
  
  std::cout << "Input Branch Config:" << std::endl;
  in_branch_cand.Print();
  
  std::cout << "Output Branch Config:" << std::endl;
  out_particles.Print();
  
  man->AddBranch(plain_branch_, out_particles);
  
  if(cuts_)
    cuts_ -> Init(*out_config);
  
  InitIndices();
  
  std::cout << "Loading ONNX model file " << model_file_name_ << std::endl;
  onnx_runner_ = new ONNXRunner();
  onnx_runner_->Init(model_file_name_);
  
  if (onnx_runner_->GetFeatureCount() != feature_field_ids_.size())
  {
    std::cout << "ERROR: ONNX Model requires " << onnx_runner_->GetFeatureCount() << " features, but " << feature_field_ids_.size() << " are given!" << std::endl;
    exit(-1);
  }
}

void ATreePredictionAdder::Exec()
{
  plain_branch_ -> ClearChannels();
  
  auto out_config = AnalysisTree::TaskManager::GetInstance()->GetConfig();
  
  // Predict probabilities with ONNX model and save to signal_probs vector
  std::vector<float> onnx_feature_values;
  for(auto& input_particle : *candidates_)
  {
    for(auto& feature_field_id : feature_field_ids_)
      onnx_feature_values.push_back(input_particle.GetField<float>(feature_field_id));
  }
  
  std::vector<float> signal_probs = onnx_runner_->PredictBatch(onnx_feature_values);
  
  // Add predictions to output candidates
  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);
  int iCandidate = -1;

  for(auto& input_particle : *candidates_)
  {
    ++iCandidate;
    if(cuts_)
      if(!cuts_->Apply(input_particle))
        continue;
    
    auto& output_particle = plain_branch_->AddChannel(out_config->GetBranchConfig(plain_branch_->GetId()));

    // Add ONNX prediction
    float signal_prob = signal_probs[iCandidate];
    output_particle.SetField(signal_prob, onnx_pred_field_id_w_);
    

    // Copy all other fields for the candidate
    for (const auto& field : in_branch_cand.GetMap<float>())
      output_particle.SetField(input_particle.GetField<float>(field.second.id_), field.second.id_);

    for (const auto& field : in_branch_cand.GetMap<int>())
      output_particle.SetField(input_particle.GetField<int>(field.second.id_), field.second.id_);

    for (const auto& field : in_branch_cand.GetMap<bool>())
      output_particle.SetField(input_particle.GetField<bool>(field.second.id_), field.second.id_);

/*    
    //for test:
    bool mcIsSignal = (input_particle.GetField<int>(generation_field_id_r_) == 1);
    
    if (mcIsSignal)
      output_particle.SetField(1.0f, onnx_pred_field_id_w_);
    else
      output_particle.SetField(0.0f, onnx_pred_field_id_w_);
*/    

  }
}

void ATreePredictionAdder::InitIndices()
{
  auto in_branch_cand = config_->GetBranchConfig(input_branch_name_);
  
  for(auto& feature_field_name : feature_field_names_)
    feature_field_ids_.push_back(in_branch_cand.GetFieldId(feature_field_name));
  
  //generation_field_id_r_       = in_branch_cand.GetFieldId("generation");
    
  auto out_config = AnalysisTree::TaskManager::GetInstance()->GetConfig();
  const auto& out_branch = out_config->GetBranchConfig(plain_branch_->GetId());
  
  onnx_pred_field_id_w_ = out_branch.GetFieldId("onnx_pred");
}

std::vector<std::string> ATreePredictionAdder::stringSplit(std::string s, std::string delimiter)
{
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos)
  {
    token = s.substr (pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back (token);
  }

  res.push_back (s.substr (pos_start));
  return res;
}

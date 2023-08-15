#include "ATreePredictionAdder.hpp"

#include "ONNXConfigManager.hpp"
#include "AnalysisTree/TaskManager.hpp"
#include "AnalysisTree/PlainTreeFiller.hpp"

int main(int argc, char **argv)
{
  std::string filename_pfs = "filelist.txt";
  std::string input_branch_name = "Candidates_plain";
  std::string output_branch_name = "Candidates_plainPredicted";
  std::string model_file_name = "model_onnx.onnx";
  std::string feature_field_names = "chi2_geo,chi2_prim_first,chi2_prim_second,distance,l_over_dl,mass2_first,mass2_second";
  std::string output_file = "prediction_tree.root";
  std::string tree_name = "pTree";
  std::string onnx_config_path = "/lustre/cbm/users/tfic/pid/onnx_20230809_154703/onnx_config.json";
  int num_threads = -1;

  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "--f") == 0)
    {
      filename_pfs = std::string(argv[++i]);
      printf("Input file: %s\n", filename_pfs.c_str());
    }
    if (strcmp(argv[i], "--ib") == 0)
    {
      input_branch_name = std::string(argv[++i]);
      printf("Input branch name: %s\n", input_branch_name.c_str());
    }
    if (strcmp(argv[i], "--ob") == 0)
    {
      output_branch_name = std::string(argv[++i]);
      printf("Output branch name: %s\n", output_branch_name.c_str());
    }
    if (strcmp(argv[i], "--m") == 0)
    {
      model_file_name = std::string(argv[++i]);
      printf("ONNX model file name: %s\n", model_file_name.c_str());
    }
    if (strcmp(argv[i], "--features") == 0)
    {
      feature_field_names = std::string(argv[++i]);
      printf("Feature field names: %s\n", feature_field_names.c_str());
    }
    if (strcmp(argv[i], "--o") == 0)
    {
      output_file = std::string(argv[++i]);
      printf("Output file: %s\n", output_file.c_str());
    }
    if (strcmp(argv[i], "--t") == 0)
    {
      tree_name = std::string(argv[++i]);
      printf("Tree name: %s\n", tree_name.c_str());
    }
    if (strcmp(argv[i], "--num_threads") == 0)
    {
      num_threads = atoi(argv[++i]);
      printf("Number of ONNX threads: %d\n", num_threads);
    }
    if (strcmp(argv[i], "--config") == 0)
    {
      onnx_config_path = atoi(argv[++i]);
      printf("Path to ONNX inferrer config: %s\n", onnx_config_path);
    }
  }

  auto *man = AnalysisTree::TaskManager::GetInstance();
  man->SetOutputName(output_file, tree_name);
  auto *at_prediction_adder_task = new ATreePredictionAdder();

  at_prediction_adder_task->SetInputBranchName(input_branch_name);
  at_prediction_adder_task->SetOutputBranchName(output_branch_name);
  at_prediction_adder_task->SetModelFileName(model_file_name);
  at_prediction_adder_task->SetFeatureFieldNames(feature_field_names);
  at_prediction_adder_task->SetNumThreads(num_threads);
  at_prediction_adder_task->SetONNXConfigPath(onnx_config_path);

  man->AddTask(at_prediction_adder_task);
  man->Init({filename_pfs}, {tree_name});
  man->Run(-1); // -1 = all events
  man->Finish();

  return EXIT_SUCCESS;
}
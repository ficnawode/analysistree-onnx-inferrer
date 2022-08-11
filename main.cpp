#include "ATreePredictionAdder.hpp"

#include "AnalysisTree/TaskManager.hpp"
#include "AnalysisTree/PlainTreeFiller.hpp"

int main(int argc, char** argv)
{

  std::string filename_pfs = "filelist.txt";
  std::string input_branch_name = "Candidates_plain";
  std::string output_branch_name = "Candidates_plainPredicted";
  std::string model_file_name = "model_onnx.onnx";
  std::string feature_field_names = "chi2_geo,chi2_prim_first,chi2_prim_second,distance,l_over_dl,mass2_first,mass2_second";
  std::string output_file = "prediction_tree.root";
  std::string tree_name = "pTree";
  
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
  }

  const bool make_plain_ttree{true};
 
  //const std::string& filename_pfs = argv[1];  
  
  auto* man = AnalysisTree::TaskManager::GetInstance();
  man->SetOutputName(output_file, tree_name);
  auto* at_prediction_adder_task = new ATreePredictionAdder();
  
  at_prediction_adder_task->SetInputBranchName(input_branch_name);
  at_prediction_adder_task->SetOutputBranchName(output_branch_name);
  at_prediction_adder_task->SetModelFileName(model_file_name);
  at_prediction_adder_task->SetFeatureFieldNames(feature_field_names);
  
  man->AddTask(at_prediction_adder_task);
  man->Init({filename_pfs}, {tree_name});
  man->Run(-1);// -1 = all events
  man->Finish();
  /*
  if(make_plain_ttree)
  {
    man->ClearTasks();
    std::ofstream filelist;
    filelist.open("filelist.txt");
    filelist << "intermediate_tree.root\n";
    filelist.close();
        
    auto* tree_task = new AnalysisTree::PlainTreeFiller();
    std::string branchname_rec = "Candidates_plain";
    tree_task->SetInputBranchNames({branchname_rec});
    tree_task->SetOutputName("analysis_plain_ttree.root", "plain_tree");
    tree_task->AddBranch(branchname_rec);

    man->AddTask(tree_task);

    man->Init({"filelist.txt"}, {"pTree"});
    man->Run(-1);// -1 = all events
    man->Finish();    
  }  
  */
  return EXIT_SUCCESS;  
}
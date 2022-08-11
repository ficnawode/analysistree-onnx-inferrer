# at_tree_prediction_adder

## General information

at_tree_prediction_adder was developed to apply predictions of of a machine learning model described in the ONNX format to an existing AnalysisTree, using the topological values stored in its branch fields as input for the model.

## Pre-requirements

Installation with c++ 17 standard is recommended.

### ONNX Runtime

ONNR Runtime is needed to compile and run the program. Get the precompiled version for your architecture here:

https://github.com/microsoft/onnxruntime/releases

After download, add the lib directory to your `$LD_LIBRARY_PATH` environment variable.

### Root

ROOT6 is needed for installation:

https://root.cern/install/build_from_source/

Follow instructions
    
### AnalysisTree

https://github.com/HeavyIonAnalysis/AnalysisTree

Follow instructions. Version since v2.2.0 is recommended.

## Installation

Clone at_tree_prediction_adder

    git clone https://git.cbm.gsi.de/apuntke/at_tree_prediction_adder.git src
    
Source ROOT

    source /path-to-root/install/bin/thisroot.sh
    
Export AnalysisTree libraries

    export AnalysisTree_DIR=/path-to-analysistree/install/lib/cmake/AnalysisTree
	
Export ONNX Runtime libraries

    export OnnxRuntime_DIR=/path-to-onnxruntime
    
Install at_tree_prediction_adder
    
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=../inst ../src
	make -j install
    
  If you use c++ 17 standard also add cmake key -DCMAKE_CXX_STANDARD=17

## Command-line arguments
### --f <input-filelist>
Specifies the input filelist.txt which is read into the program. It must contain a list of PFStimple AnalysisTree root files which should be processed.
### --ib <branch-name>
Specifies the name of the branch in the input file where the candidates for which the prediction should be made are stored in.
### --ob <branch-name>
Specifies the name of the output branch where the candidates including the prediction field should be saved. Must be different from input branch name.
### --m <onnx-file>
Specifies the *.onnx file where the model is stored in
### --features <feature-list>
Specifies the order and field names of the features which are put into the model in a comma-separated list.
### --o <output-file>
Specifies the output file name where the root tree should be stored in.
### --t
Specified the name of the tree inside the input and output file where the candidates are stored in.

# Usage example
In python, given a trained XGBClassifier `model_clf`, we can export it to the *.onnx format using the [hipe4ml converter](https://github.com/fgrosa/hipe4ml_converter) (install with `pip install hipe4ml-converter`):
```python
from hipe4ml_converter.h4ml_converter import H4MLConverter
features_for_train = ["Candidates_plain_chi2_geo", "Candidates_plain_chi2_prim_first", "Candidates_plain_chi2_prim_second", "Candidates_plain_distance", "Candidates_plain_l_over_dl", "Candidates_plain_mass2_first", "Candidates_plain_mass2_second"]
model_conv = H4MLConverter(model_clf)
model_onnx = model_conv.convert_model_onnx(len(features_for_train))
onnx_session = InferenceSession(model_onnx.SerializeToString())
model_conv.dump_model_onnx("xgboost_lambda_classifier.onnx")
```
Next we run `at_tree_prediction_adder` with a filelist containing a file generated with [PFSimple](https://github.com/HeavyIonAnalysis/PFSimple) which candidates contain all the fields the model needs:
```
./at_tree_prediction_adder -f filelist.txt --ib Candidates_plain --ob Candidates_plainPredicted --m xgboost_lambda_classifier.onnx --features chi2_geo,chi2_prim_first,chi2_prim_second,distance,l_over_dl,mass2_first,mass2_second --o prediction_tree.root
```
Then you can analyze the outcoming file `prediction_tree.root` with AnalysisTreeQA and use the new candidate field `onnx_pred` (which should contain the signal probability) to apply cuts on.
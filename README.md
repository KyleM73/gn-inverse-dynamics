# gn-inverse-dynamics

## based on graph_nets from deepmind
https://github.com/deepmind/graph_nets 

## Virtual environment settings

### Use conda
$ conda env create -f tf2-gnets.yml

### Manual pip install
$ pip install 

## code
- main.py : start trainig from random initial parameters, traning parameters are saved in "saved_model" folder
- main_load.py : start training from "saved_model" parameters
- load_test.py : check the saved parameters can be loaded well
- load_test_save_result.py : record the target & model ouput data in "results" folder


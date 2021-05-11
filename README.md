# gn-inverse-dynamics

## based on graph_nets from deepmind
https://github.com/deepmind/graph_nets 

## Virtual environment settings

### Use conda
```
$ conda env create -f tf2-gnets.yml
```

### Manual pip install
```
$ pip install --upgrade pip
$ pip install tensorflow
$ pip install dm-sonnet
$ pip install PyGeometry
$ pip install scipy
$ pip install urdfpy
```
(CPU)
```
$ pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```
(GPU)
```
$ pip install graph_nets "tensorflow_gpu>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability
```

## code
- main.py : start trainig from random initial parameters, traning parameters are saved in "saved_model" folder
- main_load.py : start training from "saved_model" parameters
- load_test.py : check the saved parameters can be loaded well
- load_test_save_result.py : record the target & model ouput data in "results" folder


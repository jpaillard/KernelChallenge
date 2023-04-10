# Kernel Methods Kaggle Challenge
[![codecov](https://codecov.io/github/jpaillard/KernelChallenge/branch/master/graph/badge.svg?token=TJZSQ80QCV)](https://codecov.io/github/jpaillard/KernelChallenge)


## Install
```
git clone <ssh/html repo>
```
Requirements and conda env
```
conda env create -f environment.yml
```

Install the package itself in developer mode
```
pip install -e .
```

## Usage
```
usage: main.py [-h] [--data_path DATA_PATH] [--n N] [--h_iter H_ITER] [--c C] [--method METHOD] [--edges]
               [--submit]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to folder that contains the dataset (.pkl files)
  --n N                 Number of samples from the dataset to use for training
  --h_iter H_ITER       Number of iterations (depth) for the WL algorithm
  --c C                 Regularization parameter for the classifier
  --method METHOD       Classifier to use (SVC or KLR)
  --edges               Use edge embedding in the WL algorithm (see report for more details)
  --submit              create submission file for the challenge
```

### The best results have been obtained using the following command line:
` 
python main.py --method SVC --n 6000 --c 0.01 --submit --h_iter 1 --edges
`
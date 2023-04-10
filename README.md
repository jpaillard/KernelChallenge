# Data Challenge - Kernel methods
[![codecov](https://codecov.io/github/jpaillard/KernelChallenge/branch/master/graph/badge.svg?token=TJZSQ80QCV)](https://codecov.io/github/jpaillard/KernelChallenge)

[https://www.kaggle.com/competitions/data-challenge-kernel-methods-2022-2023](https://www.kaggle.com/competitions/data-challenge-kernel-methods-2022-2023)


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

## Description
```
.
├── KernelChallenge       # Main package
│   ├── kernels.py        # WL kernel implementation
│   ├── SVC.py            # SVC implementation
|   ├── KLR.py            # KLR implementation
|   ├── preprocessing.py  # preprocessing scripts before WL kernel
│   └── ...
├── tests                 # PyTest scripts
│   └── ...
├── environment.yml       # Conda environment
├── format_output.py      # script to format the prediction for kaggle competition   
├── main.py               # executable script
└── ...
```

Implementation of the Weisfeiler-lehman kernel for graph classification. And two kernel methods classification algorithms: SVR and KLR. 

```BibTex
@article{shervashidze2011weisfeiler,
  title={Weisfeiler-lehman graph kernels.},
  author={Shervashidze, Nino and Schweitzer, Pascal and Van Leeuwen, Erik Jan and Mehlhorn, Kurt and Borgwardt, Karsten M},
  journal={Journal of Machine Learning Research},
  year={2011}
}
```
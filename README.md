<p align="center">
<img src="images/logo.png" width=750/>
</p>

# Symbolic DAG Search
Systematically searching the space of small directed, acyclic graphs (DAGs).

Published in the Paper: Scaling Up Unbiased Search-based Symbolic Regression, where it is named UDFS (Unbiased DAG Frame Search).

For some reason, Google Search did not pick up upon the original github repo. So if you found this repo via the github.io page, you can find the original repository [here](https://github.com/kahlmeyer94/DAG_search).

## Installation
**Option 1**

Clone the repository and then follow
```
conda create --name testenv python=3.9.12
conda activate testenv
pip install -r requirements.txt

... do stuff here

conda deactivate
conda remove -n testenv --all
```

**Option 2**

Copy the install script `install.sh` and then run
```
bash install.sh
```

## Usage

Lets consider a regression problem with `N` samples of inputs `X` (shape `N x m`) and outputs `y` (shape `N`). 

Estimation of an expression can be done with three types of regressors:

#### UDFS
This is the base UDFS regressor.

```
from DAG_search import dag_search
est = dag_search.DAGRegressor()
est.fit(X, y)
```

#### UDFS + Aug
This is UDFS with Augmentations as described in our paper.

```
from DAG_search import dag_search
est = dag_search.DAGRegressorPoly()
est.fit(X, y)
```

#### UDFS + Aug + Eliminations
Here we wrap any symbolic regressor into an outer loop that detects variable eliminations.
This is especially useful if have regression problems with a lot of inputs.

```
from DAG_search import dag_search, eliminations
regr = dag_search.DAGRegressorPoly() # or DAGRegressor
est = eliminations.EliminationRegressor(regr)
est.fit(X, y)
```

#### Inference of the models

The fitted expression can then be accessed via
```
est.model()
```
Note that the model is returned as a [sympy](https://www.sympy.org/en/index.html) expression.

For prediction simply use 
```
pred = est.predict(X)
```
or 
```
pred, grad = est.predict(X, return_grad = True)
```


For advanced usage see the Tutorial-Notebook `tutorial.ipynb`.

## Citation
To reference this work, please use the following citation:
```
@inproceedings{Kahlmeyer:IJCAI24,
  title     = {Scaling Up Unbiased Search-based Symbolic Regression},
  author    = {Kahlmeyer, Paul and Giesen, Joachim and Habeck, Michael and Voigt, Henrik},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {4264--4272},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/471},
  url       = {https://doi.org/10.24963/ijcai.2024/471},
}
```

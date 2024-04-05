<p align="center">
<img src="images/logo.png" width=750/>
</p>

# Symbolic DAG Search
Systematically searching the space of small directed, acyclic graphs (DAGs).


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

Estimation of an expression is as simple as this:

```
from DAG_search import dag_search
est = dag_search.DAGRegressor()
est.fit(X, y)
```

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



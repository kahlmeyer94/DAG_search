import sympy
import warnings
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch.optim as optim
import torch.nn as nn

'''
Black Box Regressors
'''
class Net(nn.Module):

    def __init__(self, inp_dim:int, outp_dim:int):
        '''
        A simple feed forward network.
        Feel free to change anything.

        @Params:
            inp_dim... dimension of input
            outp_dim... dimension of output
        '''

        super(Net, self).__init__()
        h_dim = 128
        n_layers = 8
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(inp_dim, h_dim))
        self.layers.append(nn.BatchNorm1d(h_dim))
        self.layers.append(nn.ELU())
        
        for _ in range(n_layers):
            self.layers.append(nn.Linear(h_dim, h_dim))
            self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(h_dim, outp_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP():
    '''
    Regressor based on Neural Net
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        
        self.net = None
        self.verbose = verbose
        self.random_state = random_state
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        assert len(y.shape) == 1
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()

        self.net = Net(X.shape[1], 1)
        inp_tensor = torch.as_tensor(self.X).float()
        outp_tensor = torch.as_tensor(self.y.reshape(-1, 1)).float()  
        

        optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        thresh = 1e-10
        n_iterations = 1000

        if self.verbose:
            pbar = tqdm(range(n_iterations))
        else:
            pbar = range(n_iterations)

        for _ in pbar:
            optimizer.zero_grad()
            pred = self.net(inp_tensor)
            loss = torch.mean((outp_tensor - pred)**2)

            loss.backward()
            optimizer.step()
            if self.verbose:
                pbar.set_postfix({'loss' : loss.item()})
            if loss.item() < thresh:
                break

    def predict(self, X):
        assert self.X is not None and self.net is not None, 'call .fit first!'
        self.net.eval()
        X_tensor = torch.as_tensor(X).float()
        with torch.no_grad():
            pred = self.net(X_tensor).detach().numpy()
        return pred.flatten()
    
class PolyReg():
    '''
    Regressor based on Polynomial Regression
    '''
    def __init__(self, degree:int = 2, verbose:int = 0, random_state:int = 0, **params):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.regr = LinearRegression()
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        X_poly = self.poly.fit_transform(self.X)
        self.regr.fit(X_poly, self.y)

    def predict(self, X):
        assert self.X is not None
        X_poly = self.poly.fit_transform(self.X)
        pred = self.regr.predict(X_poly)
        return pred

    def model(self):

        assert self.X is not None
        names = [sympy.symbols(f'x_{i}', real = True) for i in range(self.X.shape[1])]


        expr = self.regr.intercept_
        for x_name, alpha in zip(names, self.regr.coef_):
            expr += alpha*x_name
        return expr

class LinReg():
    '''
    Regressor based on Linear Regression
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        
        self.regr = LinearRegression()
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        self.regr.fit(self.X, self.y)

    def predict(self, X):
        assert self.X is not None, 'call .fit() first!'
        pred = self.regr.predict(X)
        return pred
    
    def model(self):
        assert self.X is not None
        names = [sympy.symbols(f'x_{i}', real = True) for i in range(self.X.shape[1])]
        expr = self.regr.intercept_
        for x_name, alpha in zip(names, self.regr.coef_):
            expr += alpha*x_name
        return expr

'''
Symbolic Regressors
'''

class Operon():
    '''
    Regressor based on Operon
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        import pyoperon
        from pyoperon.sklearn import SymbolicRegressor as OperonRegressor
        self.regressor_operon = OperonRegressor(random_state = random_state)
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()

        self.regressor_operon.fit(self.X, self.y)

    def predict(self, X):
        assert self.X is not None
        pred = self.regressor_operon.predict(X)
        return pred.flatten()

    def model(self):
        assert self.X is not None
        names = [f'x_{i}' for i in range(self.X.shape[1])]
        model_str = self.regressor_operon.get_model_string(self.regressor_operon.model_, names = names)
        return sympy.sympify(model_str)
    
class GPlearn():
    '''
    Regressor based on gplearn
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        import gplearn
        from gplearn.genetic import SymbolicRegressor as GPlearnRegressor

        self.converter = {
            'add': lambda x, y : x + y,
            'sub' : lambda x, y: x - y,
            'mul': lambda x, y : x * y,
            'div' : lambda x, y: x / y,
            'neg': lambda x : -x,
            'inv': lambda x : 1/x,
            'sin' : lambda x: sympy.sin(x),
            'cos' : lambda x: sympy.cos(x),
            'log' : lambda x: sympy.log(x),
            'sqrt' : lambda x: sympy.sqrt(x),
        }
        funcs = list(self.converter.keys())
        
        self.est_gp = GPlearnRegressor(function_set = funcs, verbose = verbose, random_state=random_state)
        
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        for i in range(self.X.shape[1]):
            self.converter[f'X{i}'] = sympy.symbols(f'x_{i}', real = True)
        self.est_gp.fit(self.X, self.y)

    def predict(self, X):
        assert self.X is not None
        pred = self.est_gp.predict(X)
        return pred.flatten()

    def model(self):
        assert self.X is not None
        for i in range(self.X.shape[1]):
            self.converter[f'X{i}'] = sympy.symbols(f'x_{i}', real = True)
        return sympy.sympify(str(self.est_gp._program), locals=self.converter)

class DSR():
    '''
    Regressor based on deep symbolic regression
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        from dso import DeepSymbolicRegressor

        function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "poly"]
        params = {
            "experiment" : {
                "seed" : random_state
            },
            "task": {
                "task_type" : "regression",
                "function_set" : function_set,
                "poly_optimizer_params" : {
                    "degree": 2,
                    "regressor": "dso_least_squares",
                    "regressor_params": {"n_max_terms" : 2},
                }

            },
            "policy" : {
                "max_length" : 25
            },
        }

        self.model = DeepSymbolicRegressor(params)
        
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X, self.y)

    def predict(self, X):
        assert self.X is not None
        pred = self.model.predict(X)
        return pred.flatten()

    def model(self):
        assert self.X is not None
        expr = self.model.program_.sympy_expr[0]
        x_symbs = expr.free_symbols
        symb_dict = {}
        for x in x_symbs:
            idx = int(str(x)[1:])
            symb_dict[idx] = x
            
        for i in symb_dict:
            expr = expr.subs(symb_dict[i], sympy.symbols(f'x_{i-1}', real = True))
        return expr

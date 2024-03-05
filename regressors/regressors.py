import sympy
import warnings
import numpy as np
import os
from tqdm import tqdm
import itertools
import sys
import requests

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

import torch
import torch.optim as optim
import torch.nn as nn

'''
Classical Regressors
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
        
    def fit(self, X, y, verbose=0):
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
    def __init__(self, degree:int = 2, alpha:float = 0.0, verbose:int = 0, random_state:int = 0, **params):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        if alpha == 0.0:
            self.regr = LinearRegression()
        else:
            self.regr = Ridge(alpha=alpha)

        self.X = None
        self.y = None
        
    def fit(self, X, y, verbose=0):
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
        X_poly = self.poly.fit_transform(X)
        pred = self.regr.predict(X_poly)
        return pred

    def model(self):
        assert self.X is not None
        names = [sympy.symbols(f'x_{i}', real = True) for i in range(self.X.shape[1])]

        X_idxs = np.arange(self.X.shape[1])
        X_poly = []
        for degree in range(1, self.degree+1):   
            poly_idxs = itertools.combinations_with_replacement(X_idxs, degree)
            for idxs in poly_idxs:
                prod = 1
                for i in idxs:
                    prod = prod*names[i]
                X_poly.append(prod)

        expr = self.regr.intercept_
        for x_name, alpha in zip(X_poly, self.regr.coef_):
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
        
    def fit(self, X, y, verbose=0):
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
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

class ESR():
    '''
    Regressor based on ESR
    '''
    def __init__(self, path_to_eqs : str = '', eq_dict : dict = None, max_complexity : int = 6, verbose:int = 0, random_state:int = 0, loss_thresh : float = 1e-6, **params):
        
        self.random_state = random_state
        self.max_complexity = max_complexity
        self.verbose = verbose
        self.loss_thresh = loss_thresh

        # Load Equations
        if eq_dict is None:
            self.eq_dict = {}
        else:
            self.eq_dict = eq_dict
        if len(self.eq_dict) == 0:
            compls = [int(n.split('_')[-1]) for n in os.listdir(path_to_eqs)]
            compls = [c for c in compls if c <= self.max_complexity]
            for compl in compls:
                load_path = os.path.join(path_to_eqs, f'compl_{compl}', f'unique_equations_{compl}.txt')
                if os.path.exists(load_path):
                    with open(load_path, 'r') as inf:
                        lines = inf.read().splitlines()
                    self.eq_dict[compl] = lines
        self.fn_eval = None
        self.expr_sympy = None

        self.X = None
        self.y = None
        
    def fit(self, X, y, verbose=0):
        import esr.generation.generator as generator # clear
        from esr.fitting.fit_single import single_function # 
        from esr.fitting.likelihood import MSE # 

        assert len(y.shape) == 1
        assert X.shape[1] == 1
        np.random.seed(self.random_state)

        self.X = X
        self.y = y
       
        # core math as specified here: https://zenodo.org/record/7339113
        #basis_functions = [["x", "a"], ["inv", "abs"], ["+", "*", "-", "/", "pow"]]
        basis_functions = [["x", "a"], ["inv", "abs", "exp", "sqrt"], ["+", "*", "-", "/", "pow"]]
            
        all_exprs = [] # sympy expressions
        all_fns = [] # executable functions
            
        # Make some mock data and define likelihood
        x = X[:, 0]
        yerr = np.zeros(x.shape)
        np.savetxt('data.txt', np.array([x, y, yerr]).T)
        likelihood = MSE('data.txt', '', data_dir=os.getcwd())
            
        if os.path.exists('data.txt'):
            os.remove('data.txt')
        # Make string to sympy mapping
        maxvar = 20
        x = sympy.symbols('x', real=True)
        a = sympy.symbols([f'a{i}' for i in range(maxvar)], real=True)
        d1 = {'x':x}
        d2 = {f'a{i}':a[i] for i in range(maxvar)}
        locs = {**d1, **d2}
            
        for complexity in sorted(list(self.eq_dict.keys())):
            if self.verbose > 0:
                print(f'Complexity {complexity}')
            all_models = self.eq_dict[complexity]

            # Empty arrays to store results
            best_expr = None
            best_loss = np.inf
            best_params = None

            if self.verbose > 1:
                pbar = tqdm(enumerate(all_models), total = len(all_models))
            else:
                pbar = enumerate(all_models)

            for i, fun in pbar:
                try:
                    # Change string to list
                    expr, nodes, _ = generator.string_to_node(fun, basis_functions, locs=locs, evalf=True)
                    labels = nodes.to_list(basis_functions)

                    # labels = pre order tokens
                    new_labels = [None] * len(labels)
                    for j, lab in enumerate(labels):
                        if lab == 'Mul':
                            new_labels[j] = '*'
                            labels[j] = '*'
                        elif lab == 'Add':
                            new_labels[j] = '+'
                            labels[j] = '+'
                        elif lab == 'Div':
                            new_labels[j] = '/'
                            labels[j] = '/'
                        else:
                            new_labels[j] = lab.lower()
                            labels[j] = lab.lower()
                    param_idx = [j for j, lab in enumerate(new_labels) if is_float(lab)]
                    assert len(param_idx) <= maxvar, fun
                    for k, j in enumerate(param_idx):
                        new_labels[j] = f'a{k}'
                    # new labels = cleaned version

                    s = generator.labels_to_shape(new_labels, basis_functions)


                    success, _, tree = generator.check_tree(s)
                    parents = [None] + [labels[p.parent] for p in tree[1:]]

                    # Replace floats with symbols (except exponents)
                    param_idx = [j for j, lab in enumerate(labels) if is_float(lab) and not (not parents[j] is None and parents[j].lower() =='pow')]
                    for k, j in enumerate(param_idx):
                        labels[j] = f'a{k}'
                    fstr = generator.node_to_string(0, tree, labels)


                    loss, _, params = single_function(labels, basis_functions, likelihood, verbose=False, return_params = True)
                    # subsitute params into expression

                    if loss < best_loss:
                        best_loss = loss
                        best_params = params
                        best_expr = fstr

                    if best_loss <= self.loss_thresh:
                        break

                except (ValueError, AssertionError):
                    pass

            # insert params into expression
            expr = best_expr
            expr = expr.replace('inv', '1/')
            for i in range(len(best_params)):
                if f'a{i}' in expr:
                    v_i = best_params[i]
                    if v_i >= 0:
                        expr = expr.replace(f'abs(a{i})', str(v_i))
                    else:
                        expr = expr.replace(f'abs(a{i})', str(-v_i))
                    expr = expr.replace(f'a{i}', str(v_i))
            best_expr = sympy.sympify(expr)

            # replace x with x_0
            x_symb = sympy.Symbol('x_0', real = True)
            best_expr = best_expr.subs(sympy.Symbol('x'), x_symb)   

            # create eval function
            np_func = sympy.lambdify(x_symb, best_expr, modules=["numpy"])

            all_exprs.append(best_expr)
            all_fns.append(np_func)

            if self.verbose > 0:
                print(f'Best expression: {str(best_expr)}\tLoss: {best_loss}')

            if best_loss <= self.loss_thresh:
                break

        if os.path.exists('data.txt'):
            os.remove('data.txt')

        # save best model
        mses = []
        for eval_fn in all_fns:
            
            try:
                pred = eval_fn(X[:, 0])
                mses.append(np.mean((pred - y)**2))
            except np.linalg.LinAlgError:
                mses.append(np.inf)
        best_idx = np.argmin(mses)
        self.fn_eval = all_fns[best_idx]
        self.expr_sympy = all_exprs[best_idx]
    
    def predict(self, X):
        assert self.X is not None
        assert X.shape[1] == 1
        pred = self.fn_eval(X[:, 0])
        return pred

    def model(self):
        assert self.X is not None
        return self.expr_sympy

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
        self.positives = []
        
    def fit(self, X, y, verbose=0):
        assert len(y.shape) == 1
        self.positives = np.all(X > 0, axis = 0)
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
        expr = sympy.sympify(model_str)
        for x in expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                expr = expr.subs(x, sympy.Symbol(str(x), positive = True))
        return expr
    
class GPlearn():
    '''
    Regressor based on gplearn.
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
        
        
        params = {
            'function_set' : funcs,
            'verbose' : verbose,
            'random_state' : random_state,
        }

        self.est_gp = GPlearnRegressor(**params)
        
        self.X = None
        self.y = None
        self.positives = []
        
    def fit(self, X, y, verbose=0):
        assert len(y.shape) == 1
        self.positives = np.all(X > 0, axis = 0)
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
            self.converter[f'X{i}'] = sympy.symbols(f'x_{i}', real = True, positive = self.positives[i])
        expr = sympy.sympify(str(self.est_gp._program), locals=self.converter)
        for x in expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                expr = expr.subs(x, sympy.Symbol(str(x), positive = True))
        return expr

class DSR():
    '''
    Regressor based on deep symbolic regression
    '''
    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        from dso import DeepSymbolicRegressor

        # Hyperparams for DSO
        function_set = ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sqrt", "poly"]
        params = {
            "experiment" : {
                "seed" : random_state
            },
            "task": {
                "task_type" : "regression",
                "function_set" : function_set,
                "metric" : "inv_nrmse",
                "threshold" : 0.999,
                "poly_optimizer_params" : {
                    "degree": 2,
                    "regressor": "dso_least_squares",
                    "regressor_params": {"n_max_terms" : 2},
                }

            },
            "training": {
                "n_samples": 50000,
                "batch_size": 1000,
                "verbose" : verbose,
                "n_cores_batch" : 1,
                "early_stopping": True,
            },

        }

        self.regr = DeepSymbolicRegressor(params)
        
        self.X = None
        self.y = None
        self.positives = []
        
    def fit(self, X, y, verbose=0):
        assert len(y.shape) == 1
        self.positives = np.all(X > 0, axis = 0)
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regr = self.regr.fit(self.X, self.y)

    def predict(self, X):
        assert self.X is not None
        pred = self.regr.predict(X)
        return pred.flatten()

    def model(self):
        assert self.X is not None
        expr = self.regr.program_.sympy_expr
        x_symbs = expr.free_symbols
        symb_dict = {}
        for x in x_symbs:
            idx = int(str(x)[1:])
            symb_dict[idx] = x
            
        for i in symb_dict:
            expr = expr.subs(symb_dict[i], sympy.symbols(f'x_{i-1}', real = True, positive = self.positives[i-1]))
        return expr

class Transformer():

    def __init__(self, verbose:int = 0, random_state:int = 0, **params):
        
        sys.path.insert(0, os.path.join(os.getcwd(), 'regressors', 'symbolicregression'))
        #from regressors.symbolicregression import symbolicregression, model
        from model import SymbolicTransformerRegressor
        del sys.path[0]
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.pt_model = self.load_transformer_()
        self.pt_model.beam_type = 'search' # default 'sampling'
        #self.pt_model.max_generated_output_len = 100 # default 200
        self.pt_model.beam_size = 20 # default 10
        self.pt_model.beam_early_stopping = False # default True
        self.regr = SymbolicTransformerRegressor(model=self.pt_model, rescale=True)
        self.X = None
        self.y = None
        self.positives = []

  
    def load_transformer_(self):
        model_path = os.path.join('regressors', 'symbolicregression', 'model.pt')
        model = None
        try:
            if not os.path.isfile(model_path): 
                url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
                r = requests.get(url, allow_redirects=True)
                open(model_path, 'wb').write(r.content)

            if os.name == 'nt':
                import pathlib
                temp = pathlib.PosixPath
                pathlib.PosixPath = pathlib.WindowsPath    
            model = torch.load(model_path, map_location=torch.device('cpu'))
            
        except Exception as e:
            print("ERROR: model not loaded! path was: {}".format(model_path))
            print(e)    
        
        return model

    def translate_transformer_(self):
        replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

        if self.regr.tree is None:
            print('zero')
            self.expr = sympy.sympify('0')
            self.func = lambda X: np.zeros((len(X), 1))
        elif self.regr.tree[0] is None: 
            print('zero')
            self.expr = sympy.sympify('0')
            self.func = lambda X: np.zeros((len(X), 1))
        else:
            model_list = self.regr.tree[0][0]
            if "relabed_predicted_tree" in model_list:
                tree = model_list["relabed_predicted_tree"]
            else:
                tree = model_list["predicted_tree"]
            self.func = self.regr.model.env.simplifier.tree_to_numexpr_fn(tree)
                
            model_str = tree.infix()
            for op,replace_op in replace_ops.items():
                model_str = model_str.replace(op,replace_op)
            self.expr = sympy.parse_expr(model_str)
        
        for x in self.expr.free_symbols:
            idx = int(str(x).split('_')[-1])
            if self.positives[idx]:
                self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True))
        
    def fit(self, X, y, verbose = 0):
        assert len(y.shape) == 1
        self.y = y.copy()
        self.positives = np.all(X > 0, axis = 0)

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regr.fit(self.X, self.y)
            self.translate_transformer_()

    def predict(self, X):
        assert hasattr(self, 'func')
        pred = self.func(X)[:, 0]
        return pred

    def model(self):
        assert hasattr(self, 'expr')
        return self.expr
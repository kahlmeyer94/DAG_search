# Simplifications based on AIFeynman Papers
# Paper1: https://www.science.org/doi/10.1126/sciadv.aay2631
# Paper2: https://arxiv.org/abs/2006.10782
# Code: https://github.com/SJ001/AI-Feynman

import numpy as np
import itertools as it
import math
import sympy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


from DAG_search import dag_search
from DAG_search import utils


####################
# Density Estimation
####################
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut, KFold

def get_density_function(X):
    # select bandwith using crossvalidation
    bandwidths = 10 ** np.linspace(-2, 1, 5) # 0.01 to 10 # TODO: 50
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(n_splits = 5))
    grid.fit(X)

    # estimate density
    kde = KernelDensity(kernel='gaussian', bandwidth = grid.best_params_['bandwidth']).fit(X)
    logprobs = kde.score_samples(X)
    k = int(0.05*len(logprobs))
    thresh = np.sort(logprobs)[k]
    #thresh = np.median(logprobs)
    return kde, thresh

####################
# Function Approximation
####################
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class PolyReg():
    '''
    Regressor based on Polynomial Regression
    '''
    def __init__(self, degree:int = 2, normalize:bool = True, **params):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.regr = LinearRegression()
        
        self.standardize = normalize
        self.X = None
        self.y = None
        


    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()

        if self.standardize:
            mu_X = np.mean(self.X, axis = 0)
            std_X = np.std(self.X, axis = 0)
            mu_y = np.mean(self.y)
            std_y = np.std(self.y)
            X_norm = (self.X - mu_X)/std_X
            X_poly = self.poly.fit_transform(X_norm)
        else:
            mu_y = 0
            std_y = 1
            X_poly = self.poly.fit_transform(self.X)
        y_tmp = (self.y-mu_y)/std_y
        self.regr.fit(X_poly, y_tmp)

    def predict(self, X):
        assert self.X is not None
        if self.standardize:
            mu_X = np.mean(self.X, axis = 0)
            std_X = np.std(self.X, axis = 0)
            mu_y = np.mean(self.y)
            std_y = np.std(self.y)
            X_norm = (X - mu_X)/std_X
            X_poly = self.poly.fit_transform(X_norm)
        else:
            mu_y = 0
            std_y = 1
            X_poly = self.poly.fit_transform(X)
        pred = self.regr.predict(X_poly)*std_y + mu_y
        return pred

def approximate_poly(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    polydegrees = np.arange(1, 7, 1)
    rmses = []
    for degree in polydegrees:
        f_appr = PolyReg(degree = degree)
        f_appr.fit(X_train, y_train)
        pred = f_appr.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - pred)**2))
        rmses.append(rmse)
        if rmse < 1e-8:
            break
    min_idx = np.argmin(rmses)
    f_appr = PolyReg(degree = polydegrees[min_idx])
    f_appr.fit(X, y)
    return f_appr



from scipy.interpolate import RBFInterpolator

class InterpReg():
    '''
    Regressor based on RBF Interpolation
    '''
    def __init__(self):
        self.interp = None
        self.X = None
        self.y = None
        


    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()

        self.interp = RBFInterpolator(self.X, self.y, neighbors = 1000)

    def predict(self, X):
        assert self.interp is not None
        pred = self.interp(X)
        return pred

def approximate_interp(X, y):
    f_appr = InterpReg()
    f_appr.fit(X, y)
    return f_appr



import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy

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
        h_dim = 64
        n_layers = 6
        self.layers = nn.ModuleList()
        self.layers.append(nn.BatchNorm1d(inp_dim))
        self.layers.append(nn.Linear(inp_dim, h_dim))
        self.layers.append(nn.ReLU())
        
        for _ in range(n_layers):
            self.layers.append(nn.Linear(h_dim, h_dim))
            self.layers.append(nn.ReLU())
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

        # normalization of output helps with a faster start
        self.y = y.copy()
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()

        self.net = Net(X.shape[1], 1)
        inp_tensor = torch.as_tensor(self.X).float()
        outp_tensor = torch.as_tensor(((self.y - self.y_mean)/self.y_std).reshape(-1, 1)).float()  
        
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        thresh = 1e-10
        n_iterations = 2000

        if self.verbose:
            pbar = tqdm(range(n_iterations))
        else:
            pbar = range(n_iterations)

        best_model = copy.deepcopy(self.net)
        best_loss = np.inf
        for _ in pbar:
            optimizer.zero_grad()
            pred = self.net(inp_tensor)
            loss = torch.mean((outp_tensor - pred)**2)

            loss.backward()
            optimizer.step()
            

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = copy.deepcopy(self.net)
                if self.verbose:
                    pbar.set_postfix({'loss' : loss.item()})
            if loss.item() < thresh:
                break
        self.net = best_model

    def predict(self, X):
        assert self.X is not None and self.net is not None, 'call .fit first!'
        self.net.eval()
        X_tensor = torch.as_tensor(X).float()
        with torch.no_grad():
            pred = self.net(X_tensor).detach().numpy()
        pred = pred.flatten()*self.y_std + self.y_mean
        return pred
   
def approximate_NN(X, y):
    f_appr = MLP(verbose = 2)
    f_appr.fit(X, y)
    return f_appr

####################
# Bottom-Up checks
# AI Feynman Symmetries
# symmetry checks adapted straight from here: 
# https://github.com/SJ001/AI-Feynman/blob/master/aifeynman/S_symmetry.py
####################


def get_a_values(X:np.ndarray, i:int, j:int, mode:str) -> np.ndarray:
    '''
    Estimates symmetry constants based on minimal distance to measurements.
    
    @Params:
        X... data (n_samples x n_dims)
        i... index of symmetry
        j... index of symmetry
        mode... kind of symmetry, has to be one of 'mult', 'div', 'add' or 'sub'
        
    @Returns:
        constants for symmetry for each datapoint
    '''
    
    
    
    assert mode in ['mult', 'div', 'add', 'sub'], f'Mode {mode} not supported!'
    
    a_values = []
    for p in X:
        if mode == 'mult':
            # x_i*a, x_j/a
            
            # x = x_i*a -> a = x/x_i
            a1 = X[:, i]/p[i]
            mask1 = np.ones(X.shape)*p
            mask1[:, i] = p[i]*a1 # = X[:, i]
            mask1[:, j] = X[:, j]/a1

            # x = x_j/a -> a = x_j/x
            a2 = p[j]/X[:, j]
            mask2 = np.ones(X.shape)*p
            mask2[:, i] = X[:, i]*a2
            mask2[:, j] = p[j]/a2 # = X[:, j]
            
        elif mode == 'div':
            # x_i*a, x_j*a
            
            # x = x_i*a -> a = x/x_i
            a1 = X[:, i]/p[i]
            mask1 = np.ones(X.shape)*p
            mask1[:, i] = p[i]*a1 # = X[:, i]
            mask1[:, j] = X[:, j]*a1

            # x = x_j*a -> a = x/x_j
            a2 = X[:, j]/p[j]
            mask2 = np.ones(X.shape)*p
            mask2[:, i] = X[:, i]*a2
            mask2[:, j] = p[j]*a2 # = X[:, j]
            
        elif mode == 'add':
            # x_i + a, x_j - a
            
            # x = x_i + a -> a = x - x_i
            a1 = X[:, i] - p[i]
            mask1 = np.ones(X.shape)*p
            mask1[:, i] = p[i] + a1 # = X[:, i]
            mask1[:, j] = X[:, j] - a1

            # x = x_j - a -> a = x_j - x
            a2 = p[j] - X[:, j]
            mask2 = np.ones(X.shape)*p
            mask2[:, i] = X[:, i] + a2
            mask2[:, j] = p[j] - a2 # = X[:, j]
            
        elif mode == 'sub':
            # x_i + a, x_j + a
            
            # x = x_i + a -> a = x - x_i
            a1 = X[:, i] - p[i]
            mask1 = np.ones(X.shape)*p
            mask1[:, i] = p[i] + a1 # = X[:, i]
            mask1[:, j] = X[:, j] + a1

            # x = x_j + a -> a = x - x_j
            a2 =  X[:, j] - p[j]
            mask2 = np.ones(X.shape)*p
            mask2[:, i] = X[:, i] + a2
            mask2[:, j] = p[j] + a2 # = X[:, j]


        dists1 = np.sum(np.abs(X - mask1), axis = 1)
        dists2 = np.sum(np.abs(X - mask2), axis = 1)

        i1 = np.argsort(dists1)[1]
        i2 = np.argsort(dists2)[1]

        if dists1[i1] < dists2[i2]:
            idx = i1
            a_values.append(a1[i1])
        else:
            idx = i2
            a_values.append(a2[i2])
    a_values = np.array(a_values)
    return a_values

def find_best_symmetry(X, y, f_appr, check_func):
    min_error = 1000
    best_ret = {'i' : 0, 'j' : 1, }
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i != j:
                ret = check_func(X, y, f_appr, i, j)
                if(ret['error'] is not None) and (ret['error'] < min_error):
                    min_error = ret['error']
                    best_ret = ret
                    best_ret['i'] = i
                    best_ret['j'] = j
    return best_ret
    #except Exception as e:
    #    print(e)
    #    return {}

def check_translational_symmetry_multiply(X, y, f_appr, i, j):
    # f(x, y) = f(x*y)?
    # make the shift x1->x1*a, x2->x2/a

    min_error = 1000
    a = 1.1
    #a = get_a_values(X, i, j, 'mult')
    min_amount = math.ceil(0.05*len(X))

    fact_translate = X.copy()
    fact_translate[:,i] = fact_translate[:,i] * a
    fact_translate[:,j] = fact_translate[:,j] / a
    list_errs = abs(y - f_appr.predict(fact_translate))

    if len(list_errs) > min_amount:
        error = np.median(list_errs)
    else:
        error = 2*min_error

    return {'error' : error}

def check_translational_symmetry_divide(X, y, f_appr, i, j):
    # f(x, y) = f(x/y)?
    # make the shift x1->x1*a, x2->x2*a

    min_error = 1000
    a = 1.1
    #a = get_a_values(X, i, j, 'div')
    min_amount = math.ceil(0.05*len(X))

    fact_translate = X.copy()
    fact_translate[:,i] = fact_translate[:,i] * a
    fact_translate[:,j] = fact_translate[:,j] * a
    list_errs = abs(y - f_appr.predict(fact_translate))


    if len(list_errs) > min_amount:
        error = np.median(list_errs)
    else:
        error = 2*min_error

    return {'error' : error}

def check_translational_symmetry_plus(X, y, f_appr, i, j):
    # f(x, y) = f(x+y)?
    # make the shift x1->x1+a, x2->x2-a

    min_error = 1000
    
    a = 0.5*min(np.std(X[:,i]), np.std(X[:,j]))
    #a = get_a_values(X, i, j, 'add')

    min_amount = math.ceil(0.05*len(X))

    fact_translate = X.copy()
    fact_translate[:,i] = fact_translate[:,i] + a
    fact_translate[:,j] = fact_translate[:,j] - a
    list_errs = abs(y - f_appr.predict(fact_translate))

    if len(list_errs) > min_amount:
        error = np.median(list_errs)
    else:
        error = 2*min_error
    return {'error' : error}
        
def check_translational_symmetry_minus(X, y, f_appr, i, j):
    # f(x, y) = f(x-y)?
    # make the shift x1->x1+a, x2->x2+a

    min_error = 1000
    a = 0.5*min(np.std(X[:,i]), np.std(X[:,j]))
    #a = get_a_values(X, i, j, 'sub')
    min_amount = math.ceil(0.05*len(X))

    fact_translate = X.copy()
    fact_translate[:,i] = fact_translate[:,i] + a
    fact_translate[:,j] = fact_translate[:,j] + a
    list_errs = abs(y - f_appr.predict(fact_translate))

    if len(list_errs) > min_amount:
        error = np.median(list_errs)
    else:
        error = 2*min_error
    return {'error' : error}


####################
# Top-down checks
####################
def find_best_variable(X, y, f_appr, check_func):
    #try:
    min_error = 1000
    best_ret = {}
    for i in range(0, X.shape[1], 1):
        ret = check_func(X, y, f_appr, i)

        if (ret['error'] is not None) and (ret['error'] < min_error):
            min_error = ret['error']
            best_ret = ret
            best_ret['i'] = i
    return best_ret
    #except Exception as e:
    #    print(e)
    #    return {}

def check_div_variable(X, y, f_appr, i):
    # f(x, y) = g(y)/x?
    # then d(1/f)/dx = 1/g(y) = (1/f)/x
    
    class InvF():
        def __init__(self, f):
            self.f = f
        def fit(self):
            pass
        def predict(self, X):
            return 1/self.f.predict(X)
    f_inv_appr = InvF(f_appr)
    return check_mult_variable(X, 1/y, f_inv_appr, i)
    
def check_mult_variable(X, y, f_appr, i):
    # f(x, y) = x*g(y)?
    # then df/dx = g(y) = f/x
    min_error = 1000
    min_amount = math.ceil(0.05*len(X))

    df_dx = utils.est_gradient(f_appr, X)
    
    top_idxs = np.arange(len(X))
    list_errs = abs(df_dx[:, i] - y/X[:, i])
    list_errs = list_errs[top_idxs]
            
    if len(list_errs) > min_amount:
        error = np.median(list_errs)
    else:
        error = 2*min_error
        
    return {'error' : error}

def check_add_variable(X, y, f_appr, i):
    # f(x, y) = c*x + g(y)?
    # then f'(x) = c
    # this will also detect dummy variables (c = 0)
    
    min_error = 1000
    min_amount = math.ceil(0.05*len(X))
    
    df_dx = utils.est_gradient(f_appr, X)
    
    top_idxs = np.arange(len(X))
    c = np.round(np.mean(df_dx[:, i]))

    list_errs = abs(df_dx[:, i] - c)
    list_errs = list_errs[top_idxs]
        
    if len(list_errs) > min_amount:
        error = np.median(list_errs)
    else:
        error = 2*min_error
    return {'error' : error, 'const' : c}


####################
# Simplification
####################

class Simplification():
    def __init__(self, f_appr, check_func, X = None, y = None, expr = None):
        assert hasattr(f_appr, 'predict') and hasattr(f_appr, 'fit')
        assert len(X.shape) == 2 and len(y.shape) == 1
        self.f_appr = f_appr
        self.check_func = check_func
        self.X = X
        self.y = y
        self.expr = expr
    
    def search(self):
        pass

    def undo(self, expr):
        pass 

class MultVar(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_mult_variable, X, y, expr)
        self.error = None
        self.idx = None
        
    def __repr__(self) -> str:
        return f'MultVar: idx={self.idx}'

    def search(self) -> tuple:
        res = find_best_variable(self.X, self.y, self.f_appr, self.check_func)
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idx = res['i']

            # new X = X without x_i
            new_X = np.column_stack([self.X[:, i] for i in range(self.X.shape[1]) if i != self.idx])
            # new y = y/x_i
            new_y = self.y/self.X[:, self.idx]
        else:
            new_X, new_y = None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idx is not None)
        # 1. translate indices of expression to make space for x_idx
        transl_dict = {}
        for i in range(self.idx):
            transl_dict[f'x_{i}'] = f'z_{i}'
        for i in range(self.idx, self.X.shape[1]):
            transl_dict[f'x_{i}'] = f'z_{i+1}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        # 2. add factor x_idx
        expr_str = f'x_{self.idx}*({expr_str})'
        self.expr = sympy.sympify(expr_str)
        return self.expr

class DivVar(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_div_variable, X, y, expr)
        self.error = None
        self.idx = None
        
    def __repr__(self) -> str:
        return f'DivVar: idx={self.idx}'

    def search(self):
        res = find_best_variable(self.X, self.y, self.f_appr, self.check_func)
        
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idx = res['i']

            # new X = X without x_i
            new_X = np.column_stack([self.X[:, i] for i in range(self.X.shape[1]) if i != self.idx])
            # new y = y*x_i
            new_y = self.y*self.X[:, self.idx]
        else:
            return None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idx is not None)
        # 1. translate indices of expression to make space for x_idx
        transl_dict = {}
        for i in range(self.idx):
            transl_dict[f'x_{i}'] = f'z_{i}'
        for i in range(self.idx, self.X.shape[1]):
            transl_dict[f'x_{i}'] = f'z_{i+1}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        # 2. add divide by x_idx
        expr_str = f'({expr_str})/x_{self.idx}'
        self.expr = sympy.sympify(expr_str)
        return self.expr

class AddVar(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_add_variable, X, y, expr)
        self.error = None
        self.idx = None
        self.c = None

    def __repr__(self) -> str:
        return f'AddVar: idx={self.idx}, c={self.c}'
        
    def search(self):
        res = find_best_variable(self.X, self.y, self.f_appr, self.check_func)
        
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idx = res['i']
            self.c = res['const']
            # new X = X without x_i
            new_X = np.column_stack([self.X[:, i] for i in range(self.X.shape[1]) if i != self.idx])
            # new y = y - c*x_i
            new_y = self.y - self.c*self.X[:, self.idx]
        else:
            return None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idx is not None)
        # 1. translate indices of expression to make space for x_idx
        transl_dict = {}
        for i in range(self.idx):
            transl_dict[f'x_{i}'] = f'z_{i}'
        for i in range(self.idx, self.X.shape[1]):
            transl_dict[f'x_{i}'] = f'z_{i+1}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        # 2. add summand c*x_idx
        expr_str = f'({self.c}*x_{self.idx}) + ({expr_str})'
        self.expr = sympy.sympify(expr_str)
        return self.expr

class MultSym(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_translational_symmetry_multiply, X, y, expr)
        self.error = None
        self.idxs = None

    def __repr__(self) -> str:
        return f'MultSym: idxs={self.idxs}'    
    
    def search(self):
        res = find_best_symmetry(self.X, self.y, self.f_appr, self.check_func)
        
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idxs = [res['i'], res['j']]

            # new X = X without x_i, x_j but x_i*x_j at index 0
            new_X = np.column_stack([self.X[:, self.idxs[0]]*self.X[:, self.idxs[1]]] + [self.X[:, i] for i in range(self.X.shape[1]) if i not in self.idxs])
            # new y = y
            new_y = self.y.copy()
        else:
            return None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idxs is not None)
        remain_idxs = [i for i in range(self.X.shape[1]) if i not in self.idxs]
        transl_dict = {'x_0' : f'(z_{self.idxs[0]}*z_{self.idxs[1]})'}
        for i in range(len(remain_idxs)):
            transl_dict[f'x_{i + 1}'] = f'z_{remain_idxs[i]}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        self.expr = sympy.sympify(expr_str)
        return self.expr

class DivSym(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_translational_symmetry_divide, X, y, expr)
        self.error = None
        self.idxs = None

    def __repr__(self) -> str:
        return f'DivSym: idxs={self.idxs}'

    def search(self):
        res = find_best_symmetry(self.X, self.y, self.f_appr, self.check_func)        
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idxs = [res['i'], res['j']]

            # new X = X without x_i, x_j but x_i/x_j at index 0
            new_X = np.column_stack([self.X[:, self.idxs[0]]/self.X[:, self.idxs[1]]] + [self.X[:, i] for i in range(self.X.shape[1]) if i not in self.idxs])
            # new y = y
            new_y = self.y.copy()
        else:
            return None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idxs is not None)
        remain_idxs = [i for i in range(self.X.shape[1]) if i not in self.idxs]
        transl_dict = {'x_0' : f'(z_{self.idxs[0]}/z_{self.idxs[1]})'}
        for i in range(len(remain_idxs)):
            transl_dict[f'x_{i + 1}'] = f'z_{remain_idxs[i]}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        self.expr = sympy.sympify(expr_str)
        return self.expr

class AddSym(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_translational_symmetry_plus, X, y, expr)
        self.error = None
        self.idxs = None

    def __repr__(self) -> str:
        return f'AddSym: idxs={self.idxs}' 

    def search(self):
        res = find_best_symmetry(self.X, self.y, self.f_appr, self.check_func)
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idxs = [res['i'], res['j']]

            # new X = X without x_i, x_j but x_i + x_j at index 0
            new_X = np.column_stack([self.X[:, self.idxs[0]] + self.X[:, self.idxs[1]]] + [self.X[:, i] for i in range(self.X.shape[1]) if i not in self.idxs])
            # new y = y
            new_y = self.y.copy()
        else:
            return None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idxs is not None)
        remain_idxs = [i for i in range(self.X.shape[1]) if i not in self.idxs]
        transl_dict = {'x_0' : f'(z_{self.idxs[0]}+z_{self.idxs[1]})'}
        for i in range(len(remain_idxs)):
            transl_dict[f'x_{i + 1}'] = f'z_{remain_idxs[i]}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        self.expr = sympy.sympify(expr_str)
        return self.expr

class SubSym(Simplification):
    def __init__(self, f_appr, X, y, expr = None):
        super().__init__(f_appr, check_translational_symmetry_minus, X, y, expr)
        self.error = None
        self.idxs = None

    def __repr__(self) -> str:
        return f'SubSym: idxs={self.idxs}'

    def search(self):
        res = find_best_symmetry(self.X, self.y, self.f_appr, self.check_func)
        if 'error' in res and 'i' in res:
            self.error = res['error']
            self.idxs = [res['i'], res['j']]

            # new X = X without x_i, x_j but x_i + x_j at index 0
            new_X = np.column_stack([self.X[:, self.idxs[0]] - self.X[:, self.idxs[1]]] + [self.X[:, i] for i in range(self.X.shape[1]) if i not in self.idxs])
            # new y = y
            new_y = self.y.copy()
        else:
            return None, None

        return new_X, new_y
    
    def undo(self, expr):
        assert (self.error is not None) and (self.idxs is not None)
        remain_idxs = [i for i in range(self.X.shape[1]) if i not in self.idxs]
        transl_dict = {'x_0' : f'(z_{self.idxs[0]}-z_{self.idxs[1]})'}
        for i in range(len(remain_idxs)):
            transl_dict[f'x_{i + 1}'] = f'z_{remain_idxs[i]}'
        expr_str = str(expr)
        for x in transl_dict:
            expr_str = expr_str.replace(x, transl_dict[x])
        expr_str = expr_str.replace('z_', 'x_')
        self.expr = sympy.sympify(expr_str)
        return self.expr

def find_best_simplifications(X:np.ndarray, y:np.ndarray, f_appr, verbose:int = 0):
    
    # Find best instance for each simplification type

    simpls = [MultVar(f_appr, X, y), DivVar(f_appr, X, y), AddVar(f_appr, X, y), 
              MultSym(f_appr, X, y), DivSym(f_appr, X, y), AddSym(f_appr, X, y), SubSym(f_appr, X, y)]
    
    ret = []
    for simpl in simpls:
        X_new, y_new = simpl.search()
        error = simpl.error
        if verbose > 1:
            print(f'{simpl}: {error}')
        ret.append({'simpl' : simpl, 'X' : X_new, 'y' : y_new})
    return ret

def find_simplifications(X:np.ndarray, y:np.ndarray, appr = approximate_poly, loss_fkt = None, verbose:int = 0):
    # loss_fkt... to decide how "good" a new problem is (smaller = better)

    # Find best eliminations by recursively checking whether new problem can still be approximated
    # Idea: a successful elimination should not make the problem more difficult

    Xs = [X.copy()]
    ys = [y.copy()]
    simpls = []

    current_X = X.copy()
    current_y = y.copy()

    f_appr = appr(current_X, current_y)
    if loss_fkt is not None:
        current_loss = loss_fkt(current_X, current_y)
    else:
        current_loss = np.mean((f_appr.predict(current_X) - current_y)**2)

    searching = True
    while current_X.shape[1] > 1 and searching:

        if verbose > 0:
            print(f'Loss: {current_loss}')

        simpl_res = find_best_simplifications(current_X, current_y, f_appr, verbose)  
        losses = []
        for res in simpl_res:
            next_X = res['X']
            next_y = res['y']

            if next_X is None:
                loss = np.inf
            else:
                if loss_fkt is not None:
                    loss = loss_fkt(next_X, next_y)
                else:
                    f_appr = appr(next_X, next_y)
                    loss = np.mean((f_appr.predict(next_X) - next_y)**2)
            losses.append(loss)
        min_idx = np.argmin(losses)

        current_X = simpl_res[min_idx]['X']
        current_y = simpl_res[min_idx]['y']
        simpl = simpl_res[min_idx]['simpl']
        next_loss = losses[min_idx]

        if verbose > 0:
            print(f'Next Loss: {next_loss}')
        if next_loss <= 10*current_loss:
            Xs.append(current_X)
            ys.append(current_y)
            simpls.append(simpl)

            f_appr = appr(current_X, current_y)
            current_loss = next_loss

            if verbose > 1:
                print(f'Chosen: {simpl}')
        else:
            searching = False

    return {'Xs' : Xs, 'ys' : ys, 'simpls' : simpls}
        

####################
# Comparing to ground truth
####################

class F_true():
    
    def __init__(self, expr):
        
        self.expr = sympy.sympify(expr)
        k = max([int(str(x).split('_')[1]) for x in self.expr.free_symbols])
        x_symbs = [f'x_{i}' for i in range(k+1)]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        return self.exec_func(*[X[:, i] for i in range(X.shape[1])])
    
def apply_symmetry(expr, subexpr):
    '''
    Applies symmetry substitution.
    '''
    # assumption: variables are named x_i (no curly brackets)
    subs_idx = sorted([int(str(x).split('_')[-1]) for x in subexpr.free_symbols if 'x_' in str(x)])
    z = sympy.Symbol('z_0')
    repl_expr = expr.subs({
        subexpr: z, # ident
        1/subexpr: 1/z, # inverse
        -subexpr: -z, # negate
        subexpr**2 : z**2, # square
        sympy.sqrt(subexpr) : sympy.sqrt(z), # square-root
        sympy.sin(subexpr) : sympy.sin(z), # sine
        sympy.cos(subexpr) : sympy.cos(z), # cosine
        sympy.log(subexpr) : sympy.log(z), # log
        sympy.exp(subexpr) : sympy.exp(z), # exp
    } )
    
    remain_idxs = sorted([int(str(x).split('_')[-1]) for x in repl_expr.free_symbols if 'x_' in str(x)])
    for i in subs_idx:
        assert i not in remain_idxs
    
    repl_expr = str(repl_expr)
    for i, idx in enumerate(remain_idxs):
        repl_expr = repl_expr.replace(f'x_{idx}', f'z_{i+1}')
    repl_expr = repl_expr.replace('z_', 'x_')
    
    repl_expr = sympy.sympify(repl_expr)
    return repl_expr

def apply_variable(expr, idx):
    '''
    Applies variable substitution.
    '''
    idxs = sorted([int(str(x).split('_')[-1]) for x in expr.free_symbols if 'x_' in str(x)])
    expr = str(utils.simplify(utils.round_floats(expr, 2)))
    assert f'x_{idx}' not in expr
    for i in idxs:
        if i > idx:
            expr = expr.replace(f'x_{i}', f'x_{i-1}')
    return sympy.sympify(expr)

def reduction_size(res, expr):
    '''
    Given a result of reductions and the ground truth expression, 
    calculates true reduction rate that would have been achieved with these reductions.
    @Params: 
        res... dict with keys Xs, ys, simpls (result from find_simplifications)
        expr... sympy expression of ground truth
        
    @Returns:
        reduction rate, which is a number between 0 (no reduction) and 1 (reduction to 1 dim)
        if expression has only one dimension, will return 1
    '''
    transform_errors = []

    Xs , ys = res['Xs'], res['ys']
    if len(Xs) == 0:
        return 1
    
    f_current = F_true(expr)

    early_stop = False
    for i in range(len(Xs) - 1):
        X_current, y_current = Xs[i], ys[i]
        y_current = f_current.predict(X_current)

        # perform simplification with true function
        simpl = res['simpls'][i]
        if 'Sym' in str(type(simpl)):
            res_true = simpl.check_func(X_current, y_current, f_current, simpl.idxs[0], simpl.idxs[1])
        else:
            res_true = simpl.check_func(X_current, y_current, f_current, simpl.idx)
        # adjust true function according to taken simplification
        try:
            if 'DivSym' in str(type(simpl)):
                expr = f_current.expr
                subexpr = sympy.sympify(f'x_{simpl.idxs[0]}/x_{simpl.idxs[1]}')
                expr_new = apply_symmetry(expr, subexpr)
            elif 'MultSym' in str(type(simpl)):
                expr = f_current.expr
                subexpr = sympy.sympify(f'x_{simpl.idxs[0]}*x_{simpl.idxs[1]}')
                expr_new = apply_symmetry(expr, subexpr)
            elif 'AddSym' in str(type(simpl)):
                expr = f_current.expr
                subexpr = sympy.sympify(f'x_{simpl.idxs[0]}+x_{simpl.idxs[1]}')
                expr_new = apply_symmetry(expr, subexpr)
            elif 'SubSym' in str(type(simpl)):
                expr = f_current.expr
                subexpr = sympy.sympify(f'x_{simpl.idxs[0]}-x_{simpl.idxs[1]}')
                expr_new = apply_symmetry(expr, subexpr)
            elif 'AddVar' in str(type(simpl)):
                expr = f_current.expr
                c = simpl.c
                subexpr = sympy.sympify(f'-{c}*x_{simpl.idx}')
                expr_new = apply_variable(expr + subexpr, simpl.idx)

            elif 'DivVar' in str(type(simpl)):
                expr = f_current.expr
                subexpr = sympy.sympify(f'x_{simpl.idx}')
                expr_new = apply_variable(expr * subexpr, simpl.idx)
            elif 'MultVar' in str(type(simpl)):
                expr = f_current.expr
                subexpr = sympy.sympify(f'x_{simpl.idx}')
                expr_new = apply_variable(expr / subexpr, simpl.idx)
            f_current = F_true(expr_new)
        except AssertionError:
            early_stop = True

        if early_stop:
            transform_errors = transform_errors + [1]*(len(Xs) - 1 - len(transform_errors))
            break
        else:
            transform_errors.append(res_true['error'])
    transform_errors = np.array(transform_errors)

    # true size reduction = # dim original - # true reductions
    mask = (transform_errors < 1e-8)
    return Xs[0].shape[1] - mask.sum()


####################
# Regressor
####################

class EliminationRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic Regression based on Elimination Tests

    Sklearn interface.
    '''

    def __init__(self, symb_regr, positives:list = None, expr = None, exec_func = None, approx = approximate_poly, **kwargs):
        '''
        @Params:
            symb_regr... symbolic regressor (has .fit(X, y), .predict(X), .model() function)
            positives... marks which X are strictly positive
        '''
        self.symb_regr = symb_regr
        self.positives = positives
        self.expr = expr
        self.exec_func = exec_func
        self.approx = approx

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 1):
        '''
        Fits a model on given regression data.
        @Params:
            X... input data (shape n_samples x inp_dim)
            y... output data (shape n_samples)
        '''
        assert len(y.shape) == 1, f'y must be 1-dimensional (current shape: {y.shape})'
        

        r2_thresh = 1-1e-8 # if solution is found with higher r2 score than this: early stop
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]

        self.positives = np.all(X > 0, axis = 0)

        if verbose > 0:
            print('Recursively searching for Eliminations')

        simpl_res = find_simplifications(X, y, appr = self.approx, verbose = verbose)
        Xs = simpl_res['Xs']
        ys = simpl_res['ys']
        simpls = simpl_res['simpls']

        if verbose > 0:
            print(f'Created {len(Xs)} regression problems.')

        # try all simplifications  + keep best one
        best_score = -np.inf
        for problem_idx in reversed(range(len(Xs))):
            X_tmp = Xs[problem_idx]
            y_tmp = ys[problem_idx]
            print(f'Size of problem: {X_tmp.shape[1]} (original: {X.shape[1]})')

            # solving with Symbolic regressor
            self.symb_regr.fit(X_tmp, y_tmp, verbose = verbose)
            pred = self.symb_regr.predict(X_tmp)
            score = r2_score(y_tmp, pred)

            if verbose > 0:
                print(f'Score: {score}')
            if score > best_score:
                best_score = score
                
                expr = self.symb_regr.model()
                for simpl in reversed(simpls[:problem_idx]):
                    expr = simpl.undo(expr)
                self.expr = expr
                #self.expr = utils.round_floats(expr)

                for x in self.expr.free_symbols:
                    idx = int(str(x).split('_')[-1])
                    if self.positives[idx]:
                        self.expr = self.expr.subs(x, sympy.Symbol(str(x), positive = True))

                self.exec_func = sympy.lambdify(x_symbs, self.expr)
            if score > r2_thresh:
                if verbose > 0:
                    print('Early stopping because solution has been found!')
                break

        return self

    def predict(self, X):
        assert hasattr(self, 'expr')

        if not hasattr(self, 'exec_func'):
            x_symbs = [f'x_{i}' for i in range(X.shape[1])]
            self.exec_func = sympy.lambdify(x_symbs, self.expr)
            
        return self.exec_func(*[X[:, i] for i in range(X.shape[1])])

    def complexity(self):
        '''
        Complexity of expression (number of calculations)
        '''
        assert hasattr(self, 'expr')
        return utils.tree_size(self.expr)

    def model(self):
        '''
        Evaluates symbolic expression.
        '''
        assert hasattr(self, 'expr'), 'No expression found yet. Call .fit first!'
        return self.expr


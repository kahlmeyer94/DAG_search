import numpy as np
import sympy

import sklearn
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

import warnings
from copy import deepcopy
import numbers

from DAG_search import utils
from DAG_search import dag_search
from DAG_search import comp_graph

########################################
# Loss Functions for DAG Search
########################################

def codec_coefficient(X, y, k = 1, normalize = True):
    if normalize:
        z = (X - np.mean(X, axis = 0))/np.std(X, axis = 0)
    else:
        z = X
    n = y.shape[0]
    
    r = y.argsort().argsort()
    l = n - r - 1
    denom = np.sum(l * (n-l))

    neigh = NearestNeighbors(n_neighbors= k+1 ).fit(z)
    nn = neigh.kneighbors(z, return_distance = False)
    
    num = np.sum(n * np.min(r[nn], axis = 1) - l**2)
    return 1- num/denom


########################################
# Eliminations + Translation
########################################

def translate_back(expr, transl_dict):
    '''
    Given an expression and a translation dict, reconstructs the original expression.

    @Params:
        expr... sympy expression
        transl_dict... translation dictionary, 
    '''
    if len(transl_dict) == 0:
        return expr

    idxs = sorted([int(str(x).split('_')[-1]) for x in expr.free_symbols if 'x_' in str(x)])
    x_expr = str(expr).replace('x_', 'z_')
    for i in idxs:
        x_expr = x_expr.replace(f'z_{i}', f'({transl_dict[i]})')
    y_expr = transl_dict[len(transl_dict) - 1]
    total_expr = f'g - ({y_expr})' # g is placeholder for rest of expression
    total_expr = sympy.sympify(total_expr)

    y_symb = sympy.Symbol('y')
    res = sympy.solve(total_expr, y_symb)
    assert len(res) > 0
    return [sympy.sympify(str(r).replace('g', f'({x_expr})')) for r in res]
    
def get_transl_dict(d, expr_sub, current_dict = {}):
    # given a substitution + translation, creates a new translation
    # d... dimensionality of X
    # expr_sub... substitution expression
    # orig_dict... current translation
    
    if len(current_dict) == 0:
        current_dict = {i : f'x_{i}' for i in range(d)}
        current_dict[d] = 'y'
        
        
    repl_idxs = sorted([int(str(x).split('_')[-1]) for x in expr_sub.free_symbols if 'x_' in str(x)])
    transl_dict = {}
    y_idx = d
    if y_idx in repl_idxs:
        # update translation
        remain_idxs = sorted([i for i in range(d) if i not in repl_idxs])
        for i, idx in enumerate(remain_idxs):
            transl_dict[i] = current_dict[idx]
        #print(transl_dict)
        #new_y = str(expr_s).replace(f'x_{y_idx}', f'({current_dict[y_idx]})')
        new_y = str(expr_sub).replace('x_', 'z_')
        for i in repl_idxs:
            new_y = new_y.replace(f'z_{i}', f'({current_dict[i]})')
        transl_dict[len(remain_idxs)] = new_y
    
    else:
        remain_idxs = sorted([i for i in range(d) if i not in repl_idxs and i!=y_idx])
        for i, idx in enumerate(remain_idxs):
            transl_dict[i+1] = current_dict[idx]
    
        new_x = str(expr_sub).replace('x_', 'z_')
        for i in repl_idxs:
            new_x = new_x.replace(f'z_{i}', f'({current_dict[i]})')
        transl_dict[0] = new_x
        transl_dict[len(transl_dict)] = current_dict[y_idx]
    k = sorted(list(transl_dict.keys()))
    return {i : transl_dict[i] for i in k}      


# Using DAG Search:

class Dimensional_Loss_Fkt(dag_search.DAG_Loss_fkt):
    def __init__(self, score_func, only_input:bool = False):
        '''
        Loss function for finding DAG for simplification task

        @Params:
            Here you can add any parameters that Loss function depends on.
        '''
        super().__init__()
        self.score_func = score_func
        self.only_input = only_input
        
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... time series (N x 1)
            cgraph... computational Graph
            c... array of constants (2D)

        @Returns:
            Loss for different constants
        '''
        if len(c.shape) == 2:
            r = c.shape[0]
            vec = True
        else:
            r = 1
            c = c.reshape(1, -1)
            vec = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fx = cgraph.evaluate(X, c = np.array([]))
            if np.all(np.isfinite(fx)) and np.all(np.abs(fx) < 1e5):
                expr = cgraph.evaluate_symbolic()[0]
                idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols])

                if X.shape[1]-1 in idxs and self.only_input:
                    loss = np.inf
                elif len(idxs) > 1 and len(idxs) < X.shape[1]:
                    if X.shape[1]-1 in idxs:
                        # we have a new y
                        new_X = np.column_stack([X[:, i] for i in range(X.shape[1]) if i not in idxs] + [fx])
                    else:
                        new_X = np.column_stack([fx] + [X[:, i] for i in range(X.shape[1]) if i not in idxs])
                    try:
                        loss = self.score_func(new_X[:, :-1], new_X[:, -1])
                    except ValueError:
                        loss = np.inf
                else:
                    loss = np.inf
            else:
                loss = np.inf
            losses = [loss]
        if not vec:
            return losses[0]
        else:
            return losses

def elimination_loop(X, y, score_fkt, nodes_elim = 1, verbose = 0, only_improve = False, only_input = False, processes = 1):
    # 1. define loss function
    loss_fkt = Dimensional_Loss_Fkt(score_fkt, only_input)

    # 2. loop and collect all subproblems
    X_all = np.column_stack([X, y])
    Xs = [X.copy()]
    ys = [y.copy()]
    exprs = []

    losses = [score_fkt(X, y)]
    invalid = False
    while (X_all.shape[1] >= 3) and (not invalid):
        # at least 2 (X) + 1 (y)
        params = {
            'X' : X_all,
            'n_outps' : 1,
            'loss_fkt' : loss_fkt,
            'k' : 0,
            'n_calc_nodes' : nodes_elim,
            'n_processes' : processes,
            'topk' : 1,
            'verbose' : verbose,
            'max_orders' : 10000, 
            'stop_thresh' : 1e-30
        }
        res = dag_search.exhaustive_search(**params)
    
        # store
        cgraph = res['graphs'][0]
        loss = res['losses'][0]

        if only_improve:
            invalid = loss > losses[0]
        else:
            invalid = False

        if not invalid:
            fx = cgraph.evaluate(X_all, c = np.array([]))
            expr = cgraph.evaluate_symbolic()[0]
            idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols])
            if X_all.shape[1]-1 in idxs:
                # we have a new y
                X_all = np.column_stack([X_all[:, i] for i in range(X_all.shape[1]) if i not in idxs] + [fx])
            else:
                X_all = np.column_stack([fx] + [X_all[:, i] for i in range(X_all.shape[1]) if i not in idxs])
            
            exprs.append(expr)
            Xs.append(X_all[:, :-1])
            ys.append(X_all[:, -1])
            losses.append(loss)
            

            if verbose > 0:
                print(f'Elimination: {str(expr)}\nNew Shape: {X_all.shape[1] -1}')
    return exprs, Xs, ys  

def beam_elimination_loop(X, y, score_fkt, beam_size = 2, nodes_elim = 1, verbose = 0, only_improve = False, only_input = False, processes = 1):
    loss_fkt = Dimensional_Loss_Fkt(score_fkt, only_input)
    ref_score = score_fkt(X, y) # reference score, should not be worse than this
    beam = [(X, y, [] , ref_score)] # list of tuples (X, y, exprs, score)
    best_result = beam[0] # best what we have found so far
    finished = False
    best_score = ref_score
    while not finished:
        new_beam = []

        # print beam
        if verbose > 0:
            print('Current beam:')
            for X_b, y_b, exprs, score in beam:
                print(exprs, score)

        for X_b, y_b, exprs, score in beam:
            if X_b.shape[1] > 1:
                X_all = np.column_stack([X_b, y_b])
                params = {
                    'X' : X_all,
                    'n_outps' : 1,
                    'loss_fkt' : loss_fkt,
                    'k' : 0,
                    'n_calc_nodes' : nodes_elim,
                    'n_processes' : processes,
                    'topk' : beam_size,
                    'verbose' : 0,
                    'max_orders' : 10000, 
                    'stop_thresh' : 1e-30
                }
                res = dag_search.exhaustive_search(**params)



                for loss, cgraph in zip(res['losses'], res['graphs']):
                    if only_improve:
                        invalid = loss > score
                    else:
                        invalid = loss > ref_score
                    if not invalid:
                        fx = cgraph.evaluate(X_all, c = np.array([]))
                        expr = cgraph.evaluate_symbolic()[0]
                        idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols])
                        if X_all.shape[1]-1 in idxs:
                            # we have a new y
                            X_new = np.column_stack([X_b[:, i] for i in range(X_b.shape[1]) if i not in idxs])
                            y_new = fx
                        else:
                            X_new = np.column_stack([fx] + [X_b[:, i] for i in range(X_b.shape[1]) if i not in idxs])
                            y_new = y_b.copy()

                        exprs_new = exprs + [expr]
                        new_beam.append((X_new, y_new, exprs_new , loss))  

                        if loss < best_score:
                            best_score = loss
                            best_result = new_beam[-1]

        
        if len(new_beam) == 0:
            #beam_scores = [x[-1] for x in beam]
            #best_result = beam[np.argmin(beam_scores)]
            finished = True
        else:
            new_beam_scores = [x[-1] for x in new_beam]
            take_idxs = np.argsort(new_beam_scores)[:beam_size]
            beam = [new_beam[i] for i in take_idxs] 
        

    # reenact best result
    Xs = [X.copy()]
    ys = [y.copy()]
    exprs = best_result[2]
    for expr in exprs:
        X_all = np.column_stack([Xs[-1], ys[-1]])
        fx = utils.eval_expr(expr, X_all)

        idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols])
        if X_all.shape[1]-1 in idxs:
            # we have a new y
            X_new = np.column_stack([Xs[-1][:, i] for i in range(Xs[-1].shape[1]) if i not in idxs])
            y_new = fx
        else:
            X_new = np.column_stack([fx] + [Xs[-1][:, i] for i in range(Xs[-1].shape[1]) if i not in idxs])
            y_new = ys[-1].copy()
        Xs.append(X_new)
        ys.append(y_new)
    return exprs, Xs, ys 

# Just Pairwise

def score_pointcloud(X_all, fx, idxs, score_fkt):
    if np.all(np.isfinite(fx)) and np.all(np.abs(fx) < 1e5):
        if X_all.shape[1]-1 in idxs:
            # we have a new y
            new_X = np.column_stack([X_all[:, i] for i in range(X_all.shape[1]) if i not in idxs] + [fx])
        else:
            new_X = np.column_stack([fx] + [X_all[:, i] for i in range(X_all.shape[1]) if i not in idxs])
        loss = score_fkt(new_X[:, :-1], new_X[:, -1])
    else:
        loss = np.inf
    return loss

def elimination_loop_pairwise(X, y, score_fkt, verbose = 0, only_improve = False):
    # loop and collect all subproblems
    X_all = np.column_stack([X, y])
    Xs = [X.copy()]
    ys = [y.copy()]
    exprs = []

    losses = [score_fkt(X, y)]
    invalid = False
    while (X_all.shape[1] >= 3) and (not invalid):
        # at least 2 (X) + 1 (y)
        best_loss = np.inf
        best_comb = 'x_0 + x_1'
        best_fx = X_all[:, 0] + X_all[:, 1]
        for i in range(X_all.shape[1]):
            for j in range(i+1, X_all.shape[1]):
                idxs = [i, j]
                # i + j
                fx = X_all[:, i] + X_all[:, j]
                loss = score_pointcloud(X_all, fx, idxs, score_fkt)
                if loss < best_loss:
                    best_loss = loss
                    best_comb = f'x_{i} + x_{j}'
                    best_fx = fx.copy()

                # i * j
                fx = X_all[:, i] * X_all[:, j]
                loss = score_pointcloud(X_all, fx, idxs, score_fkt)
                if loss < best_loss:
                    best_loss = loss
                    best_comb = f'x_{i} * x_{j}'
                    best_fx = fx.copy()

                # i - j
                fx = X_all[:, i] - X_all[:, j]
                loss = score_pointcloud(X_all, fx, idxs, score_fkt)
                if loss < best_loss:
                    best_loss = loss
                    best_comb = f'x_{i} - x_{j}'
                    best_fx = fx.copy()
                
                # j - i
                fx = X_all[:, j] - X_all[:, i]
                loss = score_pointcloud(X_all, fx, idxs, score_fkt)
                if loss < best_loss:
                    best_loss = loss
                    best_comb = f'x_{j} - x_{i}'
                    best_fx = fx.copy()

                # i / j
                fx = X_all[:, i] / X_all[:, j]
                loss = score_pointcloud(X_all, fx, idxs, score_fkt)
                if loss < best_loss:
                    best_loss = loss
                    best_comb = f'x_{i} / x_{j}'
                    best_fx = fx.copy()

                # j / i
                fx = X_all[:, j] / X_all[:, i]
                loss = score_pointcloud(X_all, fx, idxs, score_fkt)
                if loss < best_loss:
                    best_loss = loss
                    best_comb = f'x_{j} / x_{i}'
                    best_fx = fx.copy()

        # store
        expr = sympy.sympify(best_comb)
        loss = best_loss
        fx = best_fx

        if only_improve:
            invalid = loss > losses[0]
        else:
            invalid = False

        if not invalid:
            idxs = sorted([int(str(s).split('_')[-1]) for s in expr.free_symbols])
            if X_all.shape[1]-1 in idxs:
                # we have a new y
                X_all = np.column_stack([X_all[:, i] for i in range(X_all.shape[1]) if i not in idxs] + [fx])
            else:
                X_all = np.column_stack([fx] + [X_all[:, i] for i in range(X_all.shape[1]) if i not in idxs])
            
            exprs.append(expr)
            Xs.append(X_all[:, :-1])
            ys.append(X_all[:, -1])
            losses.append(loss)
            
            if verbose > 0:
                print(f'Elimination: {str(expr)}\nNew Shape: {X_all.shape[1] -1}')
    return exprs, Xs, ys      





########################################
# Verification with ground truth
########################################

def check_correctness(d, expr_true, expr_sub):
    
    repl_idxs = sorted([int(str(x).split('_')[-1]) for x in expr_sub.free_symbols if 'x_' in str(x)])
    y_idx = d
    try:
        if y_idx in repl_idxs:
            # z -> y
            z = str(expr_sub).replace(f'x_{y_idx}', f'({str(expr_true)})')
            z = sympy.sympify(z)
            z = utils.simplify(z)

            # if this passes, its correct
            for s in z.free_symbols:
                assert int(str(s).split('_')[-1]) not in repl_idxs

            # create new expression
            remain_idxs = sorted([i for i in range(d) if i not in repl_idxs])
            expr_new = str(z).replace('x_', 'z_')
            for i, idx in enumerate(remain_idxs):
                expr_new = expr_new.replace(f'z_{idx}', f'x_{i}')
            expr_new = sympy.sympify(expr_new)
        
        else:
            # z -> x_0
            z_symb = sympy.Symbol('z')
            repl_symb = list(expr_sub.free_symbols)[0]
            res = sympy.solve(expr_sub - z_symb, repl_symb)
            assert len(res) == 1

            z = str(expr_true).replace(str(repl_symb), f'({str(res[0])})')
            z = sympy.sympify(z)
            z = utils.simplify(z)

            # if this passes, its correct
            for i in repl_idxs:
                assert f'x_{i}' not in str(z)
            
            
            # create new expression
            # replace z with z_-1, all others with z_i
            z = z.subs(z_symb, sympy.Symbol('x_-1'))
            remain_idxs = sorted([int(str(s).split('_')[-1]) for s in z.free_symbols])
            expr_new = str(z).replace('x_', 'z_')
            for i, idx in enumerate(remain_idxs):
                expr_new = expr_new.replace(f'z_{idx}', f'x_{i}')
            expr_new = sympy.sympify(expr_new)
            
    except AssertionError:
        return (False, None)
    
    return (True, expr_new)

def get_true_red_size(Xs:list, exprs:list, expr_true, ret_expr:bool = False) -> int:
    '''
    Calculates minimum input dimension that would be true given a list of eliminations.

    @Params:
        Xs... list of input data (for each problem)
        exprs... list of sympy expressions (eliminations)
        expr_true... sympy expression for true expression
        ret_expr... if set, returns the last correct expression + problem
    '''
    expr_t = expr_true
    expr_ret = expr_t
    red_size = Xs[0].shape[1]
    for i in range(len(exprs)):
        expr_s = exprs[i]
        is_valid, expr_t = check_correctness(Xs[i].shape[1], expr_t, expr_s)
        if not is_valid:
            break
        else:
            expr_ret = expr_t
            red_size = Xs[i+1].shape[1]
    if ret_expr:
        return red_size, expr_ret
    return red_size

########################################
# Regressor
########################################

class EliminationRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic Regression based on Elimination Tests

    Sklearn interface.
    '''

    def __init__(self, symb_regr, only_input:bool = False, positives:list = None, expr = None, exec_func = None, score_fkt = codec_coefficient, early_stop_thresh = 0.99999, **kwargs):
        '''
        @Params:
            symb_regr... symbolic regressor (has .fit(X, y), .predict(X), .model() function)
            positives... marks which X are strictly positive
            score_fkt... function, that takes X, y and outputs a score (lower = better)
        '''
        self.symb_regr = symb_regr
        self.positives = positives
        self.expr = expr
        self.exec_func = exec_func
        self.score_fkt = score_fkt
        self.only_input = only_input
        self.early_stop_thresh = early_stop_thresh

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 1):
        '''
        Fits a model on given regression data.
        @Params:
            X... input data (shape n_samples x inp_dim)
            y... output data (shape n_samples)
        '''
        assert len(y.shape) == 1, f'y must be 1-dimensional (current shape: {y.shape})'
        

        r2_thresh = self.early_stop_thresh # if solution is found with higher r2 score than this: early stop
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]

        self.positives = np.all(X > 0, axis = 0)

        if verbose > 0:
            print('Recursively searching for Eliminations')

        exprs, Xs, ys = elimination_loop(X, y, only_input = self.only_input, score_fkt=self.score_fkt, verbose=verbose)
        # get translation dictionaries
        transl_dicts = [{}]
        for i in range(len(exprs)):
            expr_s = exprs[i]
            transl_dict = get_transl_dict(Xs[i].shape[1], expr_s, current_dict = transl_dicts[-1])
            transl_dicts.append(deepcopy(transl_dict))

        if verbose > 0:
            print(f'Finished!\nCreated {len(Xs)} regression problems')
            print(f'Trying to solve problems')

        # try all simplifications  + keep best one
        best_score = -np.inf
        self.expr = sympy.sympify(0)
        self.exec_func = sympy.lambdify(x_symbs, self.expr)
        for problem_idx in reversed(range(len(Xs))):
            X_tmp = Xs[problem_idx]
            y_tmp = ys[problem_idx]
            transl_dict = transl_dicts[problem_idx]
            if verbose > 0:
                print(f'Size of problem: {X_tmp.shape[1]} (original: {X.shape[1]})')

            # solving with Symbolic regressor
            self.symb_regr.fit(X_tmp, y_tmp, verbose = verbose)
            try:
                expr_list = translate_back(self.symb_regr.model(), transl_dict)
                scores = []
                exec_funcs = []
                current_exprs = []

                for current_expr in expr_list:
                    for x in current_expr.free_symbols:
                        idx = int(str(x).split('_')[-1])
                        if self.positives[idx]:
                            current_expr = current_expr.subs(x, sympy.Symbol(str(x), positive = True))
                    current_exprs.append(current_expr)
                    exec_func = sympy.lambdify(x_symbs, current_expr)
                    pred = exec_func(*[X[:, i] for i in range(X.shape[1])])
                    score = r2_score(y, pred)
                    scores.append(score)
                    exec_funcs.append(exec_func)
                best_idx = np.argmax(scores)
                score = scores[best_idx]
                exec_func = exec_funcs[best_idx]
                current_expr = current_exprs[best_idx]
            except:
                score = -np.inf
                continue

            if verbose > 0:
                print(f'Score: {score}')
            if score > best_score:
                self.expr = current_expr
                self.exec_func = exec_func
                best_score = score

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

        pred = self.exec_func(*[X[:, i] for i in range(X.shape[1])])
        if isinstance(pred, numbers.Number):
            pred = pred*np.ones(X.shape[0])
        return pred

    def complexity(self):
        '''
        Complexity of expression (number of calculations)
        '''
        assert hasattr(self, 'expr')
        return len(list(sympy.preorder_traversal(self.expr)))

    def model(self):
        '''
        Evaluates symbolic expression.
        '''
        assert hasattr(self, 'expr'), 'No expression found yet. Call .fit first!'
        return self.expr


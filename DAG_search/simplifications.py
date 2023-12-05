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
# Simplifications
####################

class Simplification():
    def __init__(self, f_appr, X = None, y = None, expr = None):
        assert hasattr(f_appr, 'predict') and hasattr(f_appr, 'fit')
        self.f_appr = f_appr
        self.X = X
        self.y = y
        self.expr = expr
    
    def find(self, X, y):
        pass

    def solve(self, exprs):
        pass

class MultSep(Simplification):
    '''
    Multiplicative Seperation for Regression Problem
    '''

    def __init__(self, f_appr, X=None, y=None, expr=None):
        super().__init__(f_appr, X, y, expr)
        self.error = None
        self.Xs = []
        self.ys = []
        self.idxs = []
        self.exprs = []

    def find(self, X, y):
        consts = np.median(X, axis = 0)*np.ones(X.shape)
        preds_all = self.f_appr.predict(X) # f(x1, x2)
        preds_consts = self.f_appr.predict(consts) # f(c1, c2)
        
        # test all subsets of indices
        all_idxs = np.arange(0, X.shape[1],1)
        min_error = np.inf
        best_i = []
        best_j = []
        
        for i in range(1, math.ceil(X.shape[1]/2) + 1):
            combs = it.combinations(all_idxs, i)
            for c in combs:
                idx_i = np.array(c)
                idx_j = np.delete(all_idxs, idx_i)
                
                v1 = X.copy() 
                v1[:, idx_j] = consts[:, idx_j] # x1, c2
                
                v2 = X.copy() 
                v2[:, idx_i] = consts[:, idx_i] # c1, x2
                
                sep_error = np.mean(np.abs(preds_all - (self.f_appr.predict(v1)*self.f_appr.predict(v2))/preds_consts)) # lower = better
                
                if sep_error < min_error:
                    min_error = sep_error
                    best_i = idx_i.copy()
                    best_j = idx_j.copy()
        
        const = np.sqrt(preds_consts[0])
        
        v1 = X.copy() 
        v1[:, best_j] = consts[:, best_j] # x1, c2
        X1 = X[:, best_i]
        y1 = self.f_appr.predict(v1)/const

        v2 = X.copy() 
        v2[:, best_i] = consts[:, best_i] # c1, x2
        X2 = X[:, best_j]
        y2 = self.f_appr.predict(v2)/const

        self.error = min_error
        self.Xs = [X1, X2]
        self.ys = [y1, y2]
        self.idxs = [best_i, best_j]
        self.exprs = []
    
    def solve(self, exprs):
        assert len(exprs) == len(self.Xs) and len(exprs) == 2
        self.exprs = exprs
        expr1 = self.exprs[0]
        expr2 = self.exprs[1]

        expr1_tmp = str(expr1)
        for i1, i2 in enumerate(self.idxs[0]):
            expr1_tmp = expr1_tmp.replace(f'x_{i1}', f'z_{i2}')
            
        expr2_tmp = str(expr2)
        for i1, i2 in enumerate(self.idxs[1]):
            expr2_tmp = expr2_tmp.replace(f'x_{i1}', f'z_{i2}')
            
        expr_new = f'({expr1_tmp})*({expr2_tmp})'
        expr_new = expr_new.replace('z_', 'x_')
        expr_new = sympy.sympify(expr_new)
        self.expr = expr_new
  
class AddSep(Simplification):
    '''
    Additive Seperation for Regression Problem
    '''

    def __init__(self, f_appr, X=None, y=None, expr=None):
        super().__init__(f_appr, X, y, expr)
        self.error = None
        self.Xs = []
        self.ys = []
        self.idxs = []
        self.exprs = []

    def find(self, X, y):
        consts = np.median(X, axis = 0)*np.ones(X.shape)
        preds_all = self.f_appr.predict(X) # f(x1, x2)
        preds_consts = self.f_appr.predict(consts) # f(c1, c2)
        
        # test all subsets of indices
        all_idxs = np.arange(0, X.shape[1],1)
        min_error = np.inf
        best_i = []
        best_j = []
        
        for i in range(1, math.ceil(X.shape[1]/2) + 1):
            combs = it.combinations(all_idxs, i)
            for c in combs:
                idx_i = np.array(c)
                idx_j = np.delete(all_idxs, idx_i)
                
                v1 = X.copy() 
                v1[:, idx_j] = consts[:, idx_j] # x1, c2
                
                v2 = X.copy() 
                v2[:, idx_i] = consts[:, idx_i] # c1, x2
                
                sep_error = np.mean(np.abs(preds_all - ((self.f_appr.predict(v1) + self.f_appr.predict(v2)) - preds_consts))) # lower = better
                
                if sep_error < min_error:
                    min_error = sep_error
                    best_i = idx_i.copy()
                    best_j = idx_j.copy()
        
        const = preds_consts[0]/2
    
        v1 = X.copy() 
        v1[:, best_j] = consts[:, best_j] # x1, c2
        X1 = X[:, best_i]
        y1 = self.f_appr.predict(v1) - const

        v2 = X.copy() 
        v2[:, best_i] = consts[:, best_i] # c1, x2
        X2 = X[:, best_j]
        y2 = self.f_appr.predict(v2)- const

        self.error = min_error
        self.Xs = [X1, X2]
        self.ys = [y1, y2]
        self.idxs = [best_i, best_j]
        self.exprs = []
    
    def solve(self, exprs):
        assert len(exprs) == len(self.Xs) and len(exprs) == 2
        self.exprs = exprs
        expr1 = self.exprs[0]
        expr2 = self.exprs[1]

        expr1_tmp = str(expr1)
        for i1, i2 in enumerate(self.idxs[0]):
            expr1_tmp = expr1_tmp.replace(f'x_{i1}', f'z_{i2}')
            
        expr2_tmp = str(expr2)
        for i1, i2 in enumerate(self.idxs[1]):
            expr2_tmp = expr2_tmp.replace(f'x_{i1}', f'z_{i2}')
            
        expr_new = f'({expr1_tmp}) + ({expr2_tmp})'
        expr_new = expr_new.replace('z_', 'x_')
        expr_new = sympy.sympify(expr_new)
        self.expr = expr_new

class SymSep(Simplification):

    def __init__(self, f_appr, X=None, y=None, expr=None, n_nodes:int = 1):
        super().__init__(f_appr, X, y, expr)
        self.error = None
        self.Xs = []
        self.ys = []
        self.transl_dict = {}
        self.exprs = []
        self.n_nodes = n_nodes

    def find(self, X, y):
        loss_fkt = dag_search.Gradient_loss_fkt(self.f_appr, X, y)
    
    
        params = {
            'X' : X,
            'n_outps' : 1,
            'loss_fkt' : loss_fkt,
            'k' : 0,
            'n_calc_nodes' : self.n_nodes,
            'n_processes' : 1,
            'topk' : 1,
            'opt_mode' : 'grid_zoom',
            'verbose' : 0,
            'max_orders' : int(1e4), 
            'stop_thresh' : 1e-20
        }
        res = dag_search.exhaustive_search(**params)
            
        loss = res['losses'][0]
        graph = res['graphs'][0]
        c = res['consts'][0]
        expr = graph.evaluate_symbolic(c = c)[0]
        I = sorted([int(str(e).split('_')[-1]) for e in expr.free_symbols if str(e).startswith('x_')])
        J = np.delete(np.arange(X.shape[1]), I)
        h_x = graph.evaluate(X, c = c)
        X1 = np.column_stack([h_x, X[:, J]])
        transl_dict = {'x_0' : str(expr).replace('x_', 'z_')}
        for i, j in enumerate(J):
            transl_dict[f'x_{i+1}'] = f'z_{j}'
        
        self.error = loss
        self.Xs = [X1]
        self.ys = [y]
        self.transl_dict = transl_dict
        self.exprs = []
 
    def solve(self, exprs):
        assert len(exprs) == 1
        self.exprs = exprs
        expr = self.exprs[0]
        tmp_expr = str(expr)
        for x_repl in self.transl_dict:
            tmp_expr = tmp_expr.replace(x_repl, f'({self.transl_dict[x_repl]})')   
        tmp_expr = tmp_expr.replace('z_', 'x_')
        
        expr_new = sympy.sympify(tmp_expr)
        self.expr = expr_new

########################
# Simplification - Tree
########################

def get_simpl(X, y):
    if X.shape[1] > 1:
        # 1. use Polynomial to fit
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        polydegrees = np.arange(1, 10, 1)
        rmses = []
        for degree in polydegrees:
            f_appr = dag_search.BaseReg(degree = degree)
            f_appr.fit(X, y)
            pred = f_appr.predict(X)
            rmse = np.sqrt(np.mean((y - pred)**2))
            rmses.append(rmse)
            if rmse < 1e-5:
                break
        min_idx = np.argmin(rmses)
        f_appr = dag_search.BaseReg(degree = polydegrees[min_idx])
        f_appr.fit(X, y)
        f_rmse = rmses[min_idx]

        # 1. Symmetry
        sym_sep = SymSep(f_appr)
        sym_sep.find(X, y)
        error = sym_sep.error        
        #print(f'Sym: {error}, {sym_sep.transl_dict["x_0"]}')
        return sym_sep
            
    return None


class SimplNode():
    def __init__(self, X, y, parent = None, children = [], simpl:Simplification = None) -> None:
        self.X = X
        self.y = y
        self.simpl = simpl
        self.simpl_str = ''
        self.parent = parent
        self.children = children
        self.expr = None
        self.child_exprs = []

    def simplify(self):
        self.simpl = get_simpl(self.X, self.y)
        if self.simpl is None:
            self.simpl_str = 'Leaf'
        else:
            if type(self.simpl) is MultSep:
                self.simpl_str = 'Mul'

            elif type(self.simpl) is AddSep:
                self.simpl_str = 'Add'

            elif type(self.simpl) is SymSep:
                self.simpl_str = 'Sym'

            self.children = [SimplNode(X_c, y_c) for X_c, y_c in zip(self.simpl.Xs, self.simpl.ys)]
            for c in self.children:
                c.parent = self
                c.simplify()
            
    def recombine(self):
        if self.expr is None:
            # leaf node
            assert len(self.children) > 0
            self.child_exprs = [c.recombine() for c in self.children]
            self.simpl.solve(self.child_exprs)
            self.expr = self.simpl.expr
        
        return self.expr


class SimplTree():
    '''
    Class for finding, solving and combining simplifications
    '''
    def __init__(self) -> None:

        pass

    def fit(self, X, y):
        self.root = SimplNode(X, y)
        self.root.simplify()
        self.tree_depth = self.depth(self.root, 0)

    def get_subproblems(self, depth):
        self.leaves = self.get_leaves(self.root, max_depth = depth)
        return [(l.X.copy(), l.y.copy()) for l in self.leaves]
        

    def depth(self, node, current_depth):
        if len(node.children) == 0:
            return current_depth
        else:
            return max([self.depth(c, current_depth+1) for c in node.children])
              
    def get_leaves(self, node, max_depth = -1):
        if len(node.children) == 0 or max_depth == 0:
            if node.expr is not None:
                return [] # already solved
            else:
                return [node]
        else:
            ret = []
            for c in node.children:
                ret += self.get_leaves(c, max_depth=max_depth-1)
            return ret
        
    def traverse(self, node, func):
        for c in node.children:
            self.traverse(c, func)
        func(node)
        
    def clear(self, node):
        node.expr = None
        for c in node.children:
            self.clear(c)    

    def combine(self, exprs):
        assert hasattr(self, 'leaves')
        assert len(self.leaves) == len(exprs)

        # set expressions at leaf nodes
        for l, expr in zip(self.leaves, exprs):
            l.expr = expr
        
        # recombine recursively
        self.expr = self.root.recombine()
        return self.expr

########################
# Simplification - Regressor
########################    

class SimplificationRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic Regression using recursive simplification

    Sklearn interface.
    '''

    def __init__(self, symb_regr, simpl_tree = None, positives:list = None, **kwargs):
        '''
        @Params:
            symb_regr... symbolic regressor (has .fit(X, y), .predict(X), .model() function)
            positives... marks which X are strictly positive
        '''
        self.symb_regr = symb_regr
        self.positives = positives
        self.simpl_tree = None
        self.expr = None
        self.exec_func = None

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 1):
        '''
        Fits a model on given regression data.
        @Params:
            X... input data (shape n_samples x inp_dim)
            y... output data (shape n_samples)
        '''
        assert len(y.shape) == 1, f'y must be 1-dimensional (current shape: {y.shape})'
        
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]

        self.positives = np.all(X > 0, axis = 0)

        if verbose > 0:
            print('Recursively searching for simplifications')
        self.simpl_tree = SimplTree()
        self.simpl_tree.fit(X, y)

        if verbose > 0:
            print('Trying to solve subproblems')

        final_scores = []
        final_exprs = []

        max_depth = self.simpl_tree.tree_depth
        for depth in range(max_depth + 1):
            self.simpl_tree.clear(self.simpl_tree.root) # remove all expressions
            subproblems = self.simpl_tree.get_subproblems(depth)
            sub_exprs = []
            for X_sub, y_sub in subproblems:
                # solve
                self.symb_regr.fit(X_sub, y_sub)
                expr = self.symb_regr.model()
                sub_exprs.append(expr)

                if verbose > 0:
                    print(sub_exprs)

            expr = self.simpl_tree.combine(sub_exprs)
            final_exprs.append(expr)

            exec_func = sympy.lambdify(x_symbs, expr)
            pred = exec_func(*[X[:, i] for i in range(X.shape[1])])
            score = np.sqrt(np.mean((y - pred)**2))
            final_scores.append(score)
            if score < 1e-10:
                break
        
        best_idx = np.argmin(final_scores)
        self.expr = final_exprs[best_idx]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)






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


import numbers
import sklearn
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import itertools
import numpy as np
import sympy
from scipy import stats
import warnings

from DAG_search import utils
from DAG_search import dag_search
from DAG_search import comp_graph

class Feature_loss_fkt(dag_search.DAG_Loss_fkt):
    def __init__(self, regr, y):
        '''
        Loss function for finding DAG for regression task.

        Measures the fit of a polynomial based on X and DAG(X).

        @Params:
            regr... regressor whos performance we compare
            y... output of regression problem (N)
        '''
        super().__init__()
        self.opt_const = False
        self.regr = regr
        self.y = y
        
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D) [not used]

        @Returns:
            1 - R2 if we use graph as new feature
        '''
        loss = np.inf
        x_repl = cgraph.evaluate(X, np.array([]))[:, 0]
        if np.all(np.isreal(x_repl) & np.isfinite(x_repl)): 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_new = np.column_stack([x_repl, X])
                try:
                    self.regr.fit(X_new, self.y)
                    pred = self.regr.predict(X_new)
                    loss = 1 - r2_score(self.y, pred)
                except (np.linalg.LinAlgError, ValueError):
                    loss = np.inf
        return loss
    
class BaseReg():
    '''
    Regressor based on Linear combination of polynomials and trigonometric functions
    '''
    def __init__(self, degree:int = 2, alpha:float = 0.0, max_terms:int = -1, interactions:int = 2, normalize:bool = False, **params):
        self.degree = degree
        #self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        if alpha <= 0.0:
            self.regr = LinearRegression()
        else:
            self.regr = Lasso(alpha = alpha, max_iter = 10000)
        self.X = None
        self.y = None
        self.x_mean = None
        self.x_std = None
        self.normalize = normalize
        self.max_terms = max_terms
        self.interactions = interactions
        
    def transform2poly(self, X):
        # polynomials up to degree
        # interactions up to interactions
        X_poly = np.column_stack([X**(i) for i in range(1, self.degree + 1)])
        X_interact = []
        idxs = np.arange(0, X.shape[1], 1)
        for combs in itertools.combinations(idxs, self.interactions):
            x_inter = np.prod(X_poly[:, combs], axis=1)
            X_interact.append(x_inter)
        if len(X_interact) > 0:
            X_interact = np.column_stack(X_interact)
            return np.column_stack([X_poly, X_interact])
        else:
            return X_poly

    def regression_p_values(self, X, y, reg):
        """
        Computes p-values using t-Test (null hyphotesis: c_i == 0)
        """

        yhat = reg.predict(X)
        X_tmp = np.column_stack([np.ones(len(X)), X])
        c_tmp = np.concatenate([np.array([reg.intercept_]), reg.coef_])
        XtX_inv = np.linalg.inv(X_tmp.T@X_tmp)

        df = len(X_tmp) - X_tmp.shape[1]
        mse = sum((y - yhat)**2)/df
        sd_err = np.sqrt(mse * XtX_inv.diagonal())
        t_vals = c_tmp/sd_err
        return 2 * (1 - stats.t.cdf(np.abs(t_vals), df))

    def fit(self, X, y):
        assert len(y.shape) == 1
        self.y = y.copy()

        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1).copy()
        else:
            self.X = X.copy()
        
        if self.normalize:
            self.x_mean = np.mean(X, axis=0)
            self.x_std = np.std(self.X, axis=0)
            self.X = (self.X-self.x_mean)/self.x_std

        X_trig = np.column_stack([np.sin(self.X), np.cos(self.X)])
        X_poly = self.transform2poly(self.X)
        X_all = np.column_stack([X_poly, X_trig])

        self.regr.fit(X_all, self.y)

        if self.max_terms > 0:
            # keep only top coefficients according to p value

            current_coef = self.regr.coef_
            
            p_values = self.regression_p_values(X_all, y, self.regr)


            supress_idxs = np.argsort(p_values[1:])[:-self.max_terms]
            current_coef[supress_idxs] = 0.0
            self.regr.coef_ = current_coef

    def predict(self, X):
        assert self.X is not None
        
        if self.normalize:
            assert self.x_std is not None and self.x_mean is not None
            X_tmp = (X - self.x_mean)/self.x_std
        else:
            X_tmp = X

        X_trig = np.column_stack([np.sin(X_tmp), np.cos(X_tmp)])
        X_poly = self.transform2poly(X_tmp)
        
        X_all = np.column_stack([X_poly, X_trig])
        
        pred = self.regr.predict(X_all)


        return pred

    def model(self):
        assert self.X is not None

        names = [sympy.symbols(f'x_{i}', real = True) for i in range(self.X.shape[1])]
        
        if self.normalize:
            assert self.x_std is not None and self.x_mean is not None
            for i in range(len(names)):
                names[i] = (names[i] - self.x_mean[i])/self.x_std[i]


        X_idxs = np.arange(self.X.shape[1])
        X_poly = []
        for degree in range(1, self.degree+1):   
            for name in names: 
                X_poly.append(name**degree)
        X_interact = []

        for combs in itertools.combinations(X_idxs, self.interactions):
            x_inter = 1.0
            for i in combs:
                x_inter = x_inter * names[i]
            X_interact.append(x_inter)

        X_trig = []
        for i in range(self.X.shape[1]):
            X_trig.append(sympy.sin(names[i]))
        for i in range(self.X.shape[1]):
            X_trig.append(sympy.cos(names[i]))
        expr = sympy.sympify(self.regr.intercept_)
        for x_name, alpha in zip(X_poly + X_interact + X_trig, self.regr.coef_):
            if abs(alpha) > 1e-10:
                expr += alpha*x_name
        return expr

class AugmentationRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Augmentation Regressor for symbolic regression.
    '''

    def __init__(self, symb_regr:sklearn.base.RegressorMixin, random_state:int = None, simpl_nodes:int = 2, topk:int = 1, max_orders:int = int(1e5), max_degree:int = 5, max_tree_size:int = 30, max_time_aug:float = 900, max_samples:int = None, processes:int = 1, fit_thresh:float = 1-(1e-8), **kwargs):
        self.random_state = random_state
        self.processes = processes
        self.regr_search = symb_regr
        self.regr_poly = None
        self.simpl_nodes = simpl_nodes
        self.max_orders = max_orders
        self.topk = topk
        self.max_degree = max_degree
        self.max_tree_size = max_tree_size
        self.max_samples = max_samples
        self.max_time_aug = max_time_aug
        self.fit_thresh = fit_thresh

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 0):
        max_tree_size = self.max_tree_size
        fit_thresh = self.fit_thresh # we consider everything above this as 'recovered'

        if self.random_state is not None:
            np.random.seed(self.random_state)
            

        # check for polynomial
        X_train, X_test, y_train, y_test = utils.split_extrapolation(X, y, test_size = 0.1, random_state = 42)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)
        polydegrees = np.arange(1, self.max_degree + 1, 1)
        found = False
        test_scores = []
        for degree in polydegrees:
            if verbose > 0:
                print(f'Testing Polynomial of degree {degree}')
            self.regr_poly = BaseReg(degree = degree)
            self.regr_poly.fit(X_train, y_train)
            pred_train = self.regr_poly.predict(X_train)
            pred_test = self.regr_poly.predict(X_test)
            s_train = r2_score(y_train, pred_train)
            s_test = r2_score(y_test, pred_test)
            test_scores.append(s_test)
            
            expr = utils.simplify(utils.round_floats(self.regr_poly.model(), round_digits=5))
            if s_train >= fit_thresh and s_test >= fit_thresh and utils.tree_size(expr) < max_tree_size:
                found = True
                break

        if found:
            # Our problem was a polynomial
            if verbose > 0:
                print('Expression is a polynomial')
            self.expr = utils.simplify(utils.round_floats(self.regr_poly.model(), round_digits=5))
            self.pareto_front = [self.expr]
        else:
            # Search for substitutions that simplify the problem
            polydegrees = np.arange(1, self.max_degree + 1, 1)
            #for degree, score in zip(polydegrees, test_scores[:len(polydegrees)]):
            #    if score > 0.999:
            #        break
            degree = polydegrees[np.argmax(test_scores[:len(polydegrees)])]
            
            if verbose > 0:
                print(f'Selected degree {degree}')
            
            self.regr_poly = BaseReg(degree = degree)
            self.regr_poly.fit(X, y)

            if verbose > 0:
                print('Searching for Replacements')

            X_sub = X
            y_sub = y
            if self.max_samples is not None:
                sub_idxs = np.arange(len(X))
                np.random.shuffle(sub_idxs)
                sub_idxs = sub_idxs[:self.max_samples]
                X_sub = X[sub_idxs]
                y_sub = y[sub_idxs]

            loss_fkt = Feature_loss_fkt(self.regr_poly, y_sub)
            params = {
                'X' : X_sub,
                'n_outps' : 1,
                'loss_fkt' : loss_fkt,
                'k' : 0,
                'n_calc_nodes' : self.simpl_nodes,
                'n_processes' : self.processes,
                'topk' : self.topk,
                'opt_mode' : 'grid_zoom',
                'verbose' : verbose,
                'max_orders' : self.max_orders, 
                'stop_thresh' : 1e-20,
                'max_time' : self.max_time_aug
            }
            res = dag_search.exhaustive_search(**params)

            # try top substitutions
            tried_subs = []
            scores = []
            exprs = []

            
            if verbose > 0:
                print('Iterating trough best replacements')

            for graph in res['graphs']:
                repl_expr = graph.evaluate_symbolic()[0]
                if verbose > 0 and not found:
                    print(f'Replacement: {repl_expr}')
                if (not found) and (str(repl_expr) not in tried_subs):
                    tried_subs.append(str(repl_expr))

                    X_new = np.column_stack([graph.evaluate(X, np.array([]))[:, 0], X])
                    if np.all(np.isfinite(X_new)):
                        if not found:
                            # fit using polynomial
                            self.regr_poly.fit(X_new, y)
                            pred = self.regr_poly.predict(X_new)
                            score = r2_score(y, pred)
                            scores.append(score)

                            expr = self.regr_poly.model()
                            expr = self._translate(X, expr, repl_expr)
                            #expr = utils.round_floats(expr, round_digits = 3)
                            expr = utils.simplify(expr)
                            exprs.append(expr)
                            ts = utils.tree_size(expr)
                            if verbose > 0:
                                print(f'Poly: {score}, Size: {ts}')
                            if score >= fit_thresh and ts < max_tree_size:
                                if verbose > 0:
                                    print(f'Expression is a polynomial on a substitution')
                                    print(expr)
                                found = True
                                break

                        if not found:
                            # fit using symbolic regressor
                            self.regr_search.fit(X_new, y, verbose = verbose)
                            pred = self.regr_search.predict(X_new)
                            score = r2_score(y, pred)
                            scores.append(score)

                            expr = self.regr_search.model()
                            expr = self._translate(X, expr, repl_expr)
                            #expr = utils.round_floats(expr, round_digits = 3)
                            expr = utils.simplify(expr)
                            exprs.append(expr)
                            ts = utils.tree_size(expr)
                            if verbose > 0:
                                print(f'Regressor: {score} Size: {ts}')
                            if score >= fit_thresh and ts < max_tree_size:
                                if verbose > 0:
                                    print(f'Expression found trough exhaustive search with a substitution')
                                    print(expr)
                                found = True
                                break
                
            if not found:
                # fit original problem using symbolic regressor
                if verbose > 0:
                    print(f'Replacement: Original Problem')
                self.regr_search.fit(X, y, verbose = verbose)
                pred = self.regr_search.predict(X)
                score = r2_score(y, pred)
                scores.append(score)
                #expr = utils.round_floats(self.regr_search.model(), round_digits = 5)
                exprs.append(expr)


            scores = np.array(scores)
            if scores[-1] >= fit_thresh:
                # take optimal model
                self.expr = exprs[-1]
                self.pareto_front = [self.expr]
            else:
                '''
                # get pareto front from rankings
                '''
                sizes = np.array([utils.tree_size(expr) for expr in exprs])
                ranks1 = np.argsort(sizes)
                ranks2 = np.argsort(-scores)

                pareto_idxs = [ranks1[0]] 
                current_score = scores[ranks1[0]] # have to be greater than this
                for i in ranks1[1:]:
                    if scores[i] > current_score:
                        pareto_idxs.append(i)
                        current_score = scores[i]

                self.pareto_front = [exprs[i] for i in pareto_idxs]

                    
                if np.any(sizes < max_tree_size):
                    # select best model of the smallest models
                    idxs = np.where(sizes < max_tree_size)[0]
                    exprs = [exprs[i] for i in idxs]
                    scores = scores[idxs]
                    self.expr = exprs[np.argmax(scores)]
                else:
                    self.expr = exprs[np.argmin(sizes)]
        

        # for equality check, make sure that variables have the right properties
        positives = np.all(X > 0, axis = 0)
        transl_dict = {}
        for s in self.expr.free_symbols:
            idx = int(str(s).split('_')[-1])
            if positives[idx]:
                transl_dict[s] = sympy.Symbol(f'x_{idx}', real = True, positive = True)
            else:
                transl_dict[s] = sympy.Symbol(f'x_{idx}', real = True)
        self.expr = self.expr.subs(transl_dict)
        
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)
         
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
        return utils.tree_size(self.expr)

    def model(self):
        '''
        Evaluates symbolic expression.
        '''
        assert hasattr(self, 'expr'), 'No expression found yet. Call .fit first!'
        return self.expr
    
    def _translate(self, X, expr, repl_expr):
        '''
        Translates the expression back.

        @Params:
            X... original data
            expr... final expression for X_new
            repl_expr... expression for replacement

        @Returns:
            translated expression
        '''

        orig_idx = np.arange(X.shape[1])
        new_idx = np.concatenate([np.array([-1]), orig_idx])
        transl_dict = {}
        for i in range(len(new_idx)):
            n_i = new_idx[i]
            if n_i == -1:
                transl_dict[i] = str(repl_expr).replace('x_', 'z_')
            else:
                transl_dict[i] = f'z_{n_i}'

        expr = str(expr)   
        for i in transl_dict:
            expr = expr.replace(f'x_{i}', f'({transl_dict[i]})')
        expr = expr.replace('z_', 'x_')
        expr = sympy.sympify(expr)

        transl_dict = {}
        for s in expr.free_symbols:
            transl_dict[s] = sympy.Symbol(str(s), real = True)
        return expr.subs(transl_dict)

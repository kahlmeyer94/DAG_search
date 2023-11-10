'''
Operations for combining computational graphs
'''
import numpy as np
import itertools
import warnings
import sympy
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle
import multiprocessing
from copy import deepcopy
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from DAG_search import config
from DAG_search import comp_graph
from DAG_search import utils


# TODO: remove
from timeit import default_timer as timer

########################
# Loss Functions + Optimizing constants
########################

class DAG_Loss_fkt(object):
    '''
    Abstract class for Loss function
    '''
    def __init__(self, opt_const:bool = True):
        self.opt_const = opt_const
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D)

        @Returns:
            Loss function for different constants
        '''
        pass

## Symbolic Regression

class MSE_loss_fkt(DAG_Loss_fkt):
    def __init__(self, outp:np.ndarray):
        '''
        Loss function for finding DAG for regression task.

        @Params:
            outp... output that DAG should match (N x n)
        '''
        super().__init__()
        self.outp = outp
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D)

        @Returns:
            MSE of graph output and desired output for different constants
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
            
            pred = cgraph.evaluate(X, c = c)
            losses = np.mean((pred.reshape(r, -1) - self.outp.flatten())**2, axis=-1)
            
            # must not be nan or inf
            invalid = ~np.isfinite(losses)
            
        # consider not using inf, since optimizers struggle with this
        losses[invalid] = 1000
        losses[losses > 1000] = 1000
        losses[losses < 0] = 1000

        if not vec:
            return losses[0]
        else:
            return losses

class R2_loss_fkt(DAG_Loss_fkt):
    def __init__(self, outp:np.ndarray):
        '''
        Loss function for finding DAG for regression task.

        @Params:
            outp... output that DAG should match (N x n)
        '''
        super().__init__()
        self.outp = outp # N x n
        self.outp_var = np.var(self.outp, axis = 0) # n
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D)

        @Returns:
            1-R2 of graph output and desired output for different constants
            = Fraction of variance unexplained
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
            
            pred = cgraph.evaluate(X, c = c) # r x N x n
            mses = np.mean((pred - self.outp)**2, axis=1) # r x n
            losses = np.mean(mses/self.outp_var, axis = 1) # r
            
            # must not be nan or inf
            invalid = ~np.isfinite(losses)
            
        # consider not using inf, since optimizers struggle with this
        losses[invalid] = 1000
        losses[losses > 1000] = 1000
        losses[losses < 0] = 1000

        if not vec:
            return losses[0]
        else:
            return losses

class R2_loss_fkt_old(DAG_Loss_fkt):
    def __init__(self, outp:np.ndarray):
        '''
        Loss function for finding DAG for regression task.

        @Params:
            outp... output that DAG should match (N x n)
        '''
        super().__init__()
        self.outp = outp
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D)

        @Returns:
            1-R2 of graph output and desired output for different constants
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
            
            pred = cgraph.evaluate(X, c = c)

            # pred: r x N x n, outp: N x n
            losses = []
            for i in range(r):
                v_pred = pred[i]
                if np.all(np.isreal(v_pred)) and np.all(np.isfinite(v_pred)):
                    avg_r2 = np.mean([r2_score(self.outp[:, j], v_pred[:, j]) for j in range(self.outp.shape[1])])
                else:
                    avg_r2 = 1000
                losses.append(1 - avg_r2)
            losses = np.array(losses)

            
            # must not be nan or inf
            invalid = np.zeros(r).astype(bool)
            invalid = invalid | np.isnan(losses)
            invalid = invalid | np.isinf(losses)
            
        # consider not using inf, since optimizers struggle with this
        losses[invalid] = 1000
        losses[losses > 1000] = 1000
        losses[losses < 0] = 1000

        if not vec:
            return losses[0]
        else:
            return losses

## Substitution

class Repl_loss_fkt(DAG_Loss_fkt):
    def __init__(self, regr, y, test_perc = 0.1):
        '''
        Loss function for finding DAG for regression task.

        @Params:
            regr... regressor whos performance we compare
            y... output of regression problem (N)
        '''
        super().__init__()
        self.opt_const = False
        self.regr = regr
        self.y = y
        
        self.test_perc = test_perc
        self.test_amount = int(self.test_perc*len(y))
        assert self.test_amount > 0, f'Too little data for test share of {self.test_perc}'
        all_idxs = np.arange(len(self.y))
        np.random.shuffle(all_idxs)
        self.test_idxs = all_idxs[:self.test_amount]
        self.train_idxs = all_idxs[self.test_amount:]
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray, return_idx:bool = False) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D) [not used]

        @Returns:
            1 - R2 if we use graph as replacement feature
        '''
        ret_idx = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            expr = cgraph.evaluate_symbolic()[0]
            used_idxs = sorted([int(str(e).split('_')[-1]) for e in expr.free_symbols])
            
            if len(used_idxs) >= 1:
                x_repl = cgraph.evaluate(X, np.array([]))[:, 0]
                if np.all(np.isreal(x_repl) & np.isfinite(x_repl)): 
                    losses = []
                    combs = []
                    for i in range(1, len(used_idxs) + 1):
                        combs += [list(x) for x in itertools.combinations(used_idxs, i)]
                    for repl_comb in combs:
                        X_new = np.column_stack([x_repl, np.delete(X, repl_comb, axis = 1)])
                        try:
                            self.regr.fit(X_new[self.train_idxs], self.y[self.train_idxs])
                            pred = self.regr.predict(X_new[self.test_idxs])
                            loss = 1 - r2_score(self.y[self.test_idxs], pred)
                        except (np.linalg.LinAlgError, ValueError):
                            loss = np.inf
                        losses.append(loss)
                    loss = np.min(losses)
                    ret_idx = combs[np.argmin(losses)]
                else:
                    loss = np.inf
            else:
                loss = np.inf

        if return_idx:
            return ret_idx
        else:
            return loss

class Fit_loss_fkt(DAG_Loss_fkt):
    def __init__(self, regr, y, test_perc = 0.2):
        '''
        Loss function for finding DAG for regression task.

        @Params:
            regr... regressor whos performance we compare
            y... output of regression problem (N)
        '''
        super().__init__()
        self.opt_const = False
        self.regr = regr
        self.y = y
        
        self.test_perc = test_perc
        self.test_amount = int(self.test_perc*len(y))
        assert self.test_amount > 0, f'Too little data for test share of {self.test_perc}'
        all_idxs = np.arange(len(self.y))
        np.random.shuffle(all_idxs)
        self.test_idxs = all_idxs[:self.test_amount]
        self.train_idxs = all_idxs[self.test_amount:]


        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D) [not used]

        @Returns:
            1 - R2 if we use graph as dimensionality reduction
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            expr = cgraph.evaluate_symbolic()[0]
            used_idxs = sorted([int(str(e).split('_')[-1]) for e in expr.free_symbols])
            
            if len(used_idxs) > 1:
                X_new = cgraph.evaluate(X, np.array([]))
                X_new = np.column_stack([X_new] + [X[:, i] for i in range(X.shape[1]) if i not in used_idxs])

                if np.all(np.isreal(X_new) & np.isfinite(X_new) & (np.abs(X_new) < 1000)): 
                    try:
                        self.regr.fit(X_new[self.train_idxs], self.y[self.train_idxs])
                        pred = self.regr.predict(X_new[self.test_idxs])
                        loss = 1 - r2_score(self.y[self.test_idxs], pred)
                    except np.linalg.LinAlgError:
                        loss = np.inf
                else:
                    loss = np.inf
            else:
                loss = np.inf
        return loss

class Gradient_loss_fkt(DAG_Loss_fkt):
    def __init__(self, regr, X, y):
        '''
        Loss function for finding a good substitution.

        @Params:
            regr... regressor to estimate gradients
            X... input of regression problem (N x m)
            y... output of regression problem (N)
            max_samples... maximum samples for estimating gradient
        '''
        super().__init__()

        self.opt_const = False
        self.regr = regr
        self.y = y
        self.regr.fit(X, y)
        self.df_dx = utils.est_gradient(self.regr, X)
        
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D) [not used]

        @Returns:
            Median absolute error of gradients for substituted variables if we use graph as substitution
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            expr = cgraph.evaluate_symbolic()[0]
            used_idxs = sorted([int(str(e).split('_')[-1]) for e in expr.free_symbols])
            
            if len(used_idxs) > 1:

                X_new, grad_new = cgraph.evaluate(X, np.array([]), return_grad = True)
                X_new = np.column_stack([X_new] + [X[:, i] for i in range(X.shape[1]) if i not in used_idxs])

                if np.all(np.isreal(X_new) & np.isfinite(X_new) & (np.abs(X_new) < 1000)): 
                    # gradient of substitution
                    dz_dx = grad_new[0]

                    # gradient of new regression problem
                    self.regr.fit(X_new, self.y) # fit on all

                    df_dz = utils.est_gradient(self.regr, X_new)[:, 0]

                    # condition
                    rhs = (dz_dx[:, used_idxs].T * df_dz).T
                    lhs = self.df_dx[:, used_idxs]

                    loss = np.median(np.abs(rhs - lhs))

                else:
                    loss = np.inf
            else:
                loss = np.inf
            
        return loss


def get_consts_grid(cgraph:comp_graph.CompGraph, X:np.ndarray, loss_fkt:DAG_Loss_fkt, c_init:np.ndarray = 0, interval_size:float = 2.0, n_steps:int = 51, return_arg:bool = False) -> tuple:
    '''
    Given a computational graph, optimizes for constants using grid search.

    @Params:
        cgraph... computational graph
        X... input for DAG
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        c_start... if given, start point for optimization
        max_it... maximum number of retries
        c_init... initial constants
        interval_size... size of search interval around c_init
    @Returns:
        constants that have lowest loss, loss
    '''
    k = cgraph.n_consts
    if k == 0:
        consts = np.array([])
        loss = loss_fkt(X, cgraph, consts)
        return consts, loss

    if not (type(c_init) is np.ndarray):
        c_init = c_init*np.ones(k)

    l = interval_size/2
    values = np.linspace(-l, l, n_steps)
    tmp = np.meshgrid(*[values]*k)
    consts = np.column_stack([x.flatten() for x in tmp])
    consts = consts + np.stack([c_init]*len(consts))

    losses = loss_fkt(X, cgraph, consts)

    best_idx = np.argmin(losses)
    return consts[best_idx], losses[best_idx]

def get_consts_grid_zoom(cgraph:comp_graph.CompGraph, X:np.ndarray, loss_fkt:DAG_Loss_fkt, interval_lower:float = -1, interval_upper:float = 1, n_steps:int = 21, n_zooms:int = 2) -> tuple:
    '''
    Given a computational graph, optimizes for constants using grid search with zooming.

    @Params:
        cgraph... computational graph
        X... input for DAG
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        c_start... if given, start point for optimization
        max_it... maximum number of retries
        interval_lower... minimum value for initial constants
        interval_upper... maximum value for initial constants

    @Returns:
        constants that have lowest loss, loss
    '''
    
    k = cgraph.n_consts
    interval_size = interval_upper - interval_lower
    c = (interval_upper + interval_lower)/2*np.ones(k)
    stepsize = interval_size/(n_steps - 1)
    for zoom in range(n_zooms):
        c, loss = get_consts_grid(cgraph, X, loss_fkt, c_init=c, interval_size = interval_size, n_steps=n_steps)
        interval_size = 2*stepsize
        stepsize = interval_size/(n_steps - 1)

    return c, loss

def get_consts_opt(cgraph:comp_graph.CompGraph, X:np.ndarray, loss_fkt:DAG_Loss_fkt, c_start:np.ndarray = None, max_it:int = 5, interval_lower:float = -1, interval_upper:float = 1) -> tuple:
    '''
    Given a computational graph, optimizes for constants using scipy.

    @Params:
        cgraph... computational graph
        X... input for DAG
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        c_start... if given, start point for optimization
        max_it... maximum number of retries
        interval_lower... minimum value for initial constants
        interval_upper... maximum value for initial constants

    @Returns:
        constants that have lowest loss, loss
    '''

    n_constants = cgraph.n_consts
    
    options = {'maxiter' : 20}
    def opt_func(c):
        return loss_fkt(X, cgraph, np.reshape(c, (1, -1)))[0]
    
    if n_constants > 0:
        success = False
        it = 0
        best_c = np.zeros(n_constants)

        if c_start is not None:
            best_c = c_start
            it = max_it - 1

        best_loss = opt_func(best_c)
        while (not success) and (it < max_it):
            it += 1
            if c_start is not None:
                x0 = c_start
            else:
                x0 = np.random.rand(n_constants)*(interval_upper - interval_lower) + interval_lower
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(fun = opt_func, x0 = x0, method = 'BFGS', options = options)
            success = res['success'] or (res['fun'] < best_loss)
        if success:
            c = res['x']
        else:
            c = best_c
    else:
        c = np.array([])
        
    return c, opt_func(c)

def get_consts_pool(cgraph:comp_graph.CompGraph, X:np.ndarray, loss_fkt:DAG_Loss_fkt, pool:list = config.CONST_POOL) -> tuple:
    '''
    Given a computational graph, optimizes for constants using a fixed pool of constants.

    @Params:
        cgraph... computational graph
        X... input for DAG
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        pool... list of constants

    @Returns:
        constants that have lowest loss, loss
    '''

    k = cgraph.n_consts

    if k > 0:
        #c_combs = itertools.permutations(pool, r = k)
        c_combs = np.stack([np.array(c) for c in itertools.combinations(pool, r = k)])
        losses = loss_fkt(X, cgraph, c_combs)

        best_idx = np.argmin(losses)
        best_loss = losses[best_idx]
        best_c = c_combs[best_idx]
    
        return best_c, best_loss
    return np.array([]), loss_fkt(X, cgraph, np.array([]))

########################
# DAG creation
########################

def get_pre_order(order:list, node:int, inp_nodes:list, inter_nodes:list, outp_nodes:list) -> tuple:
    '''
    Given a DAG creation order, returns the pre order of the subtree with a given node as root.
    Only used internally by get_build_orders.
    @Params:
        order... list of parents for each node (2 successive entries = 1 node)
        node... node in order list. This node will be root
        inp_nodes... list of nodes that are input nodes
        inter_nodes... list of nodes that are intermediate nodes
        out_nodes... list of nodes that are output nodes

    @Returns:
        preorder as tuple
    '''

    if node in outp_nodes:
        idx = 2*len(inter_nodes) + 2*(node-len(inp_nodes))
    else:
        idx = 2*(node - len(inp_nodes) - len(outp_nodes))
    idx_l = idx
    idx_r = idx + 1
    
    v_l = order[idx_l]
    v_r = order[idx_r]
    
    if (v_l in inp_nodes):
        return (node, v_l, v_r)
    elif (v_r in inp_nodes) or (v_r < 0):
        return (node,) + get_pre_order(order, v_l, inp_nodes, inter_nodes, outp_nodes) + (v_r,)
    else:
        return (node, ) +get_pre_order(order, v_l, inp_nodes, inter_nodes, outp_nodes) + get_pre_order(order, v_r, inp_nodes, inter_nodes, outp_nodes)

def build_dag(build_order:list, node_ops:list, m:int, n:int, k:int) -> comp_graph.CompGraph:
    '''
    Given a build order, builds a computational DAG.

    @Params:
        build_order... list of tuples (node, parent_nodes)
        node_ops... list of operations for each node
        m... number of input nodes
        n... number of output nodes
        k... number of constant nodes

    @Returns:
        Computational DAG
    '''

    node_dict = {}
    for i in range(m):
        node_dict[i] = ([], 'inp')
    for i in range(k):
        node_dict[i + m] = ([], 'const')

    for op, (i, parents) in zip(node_ops, build_order):
        node_dict[i] = (list(parents), op)
    return comp_graph.CompGraph(m = m, n = n, k = k, node_dict = node_dict)

def adapt_ops(cgraph:comp_graph.CompGraph, build_order:list, node_ops:list) -> comp_graph.CompGraph:
    '''
    Given a computational Graph, changes the operations at nodes (no need to reinstantiate)

    @Params:
        cgraph... Computational DAG
        build_order... list of tuples (node, parent_nodes)
        node_ops... list of operations for each node

    @Returns:
        Computational DAG
    '''
    node_dict = cgraph.node_dict
    for op, (i, parents) in zip(node_ops, build_order):
        node_dict[i] = (list(parents), op)
    cgraph.node_dict = node_dict
    return cgraph

def get_build_orders(m:int, n:int, k:int, n_calc_nodes:int, max_orders:int = 10000, verbose:int = 0, fix_size : bool = False) -> list:
    '''
    Creates empty DAG scaffolds (no operations yet).

    @Params:
        m... number of input nodes
        n... number of output nodes
        k... number of constant nodes
        n_calc_nodes... number of intermediate nodes
        max_orders... maximum number of possible DAG orders to search trough (lower = exhaustive, higher = sampling)
        verbose... 0 - no print, 1 - status message, 2 - progress bar
        fix_size... if set, will return only build orders with n_calc_nodes intermediate nodes (not less)

    @Returns:
        list build orders (can be used by build_dag).
        build order = list of tuples (node, parent_nodes)
    '''

    if verbose > 0:
        print('Creating evaluation orders')
    l = n_calc_nodes
    inp_nodes = [i for i in range(m + k)]
    outp_nodes = [i + m + k for i in range(n)]
    inter_nodes = [i + m + k + n for i in range(l)]

    # collect possible predecessors
    predecs = {}
    for i in inp_nodes:
        predecs[i] = []

    for i in outp_nodes:
        predecs[i] = inp_nodes + inter_nodes

    for i in inter_nodes:
        predecs[i] = inp_nodes + [j for j in inter_nodes if j < i]

    # create sample space for edges
    sample_space_edges = []
    for i in inter_nodes + outp_nodes:
        sample_space_edges.append(predecs[i])
        sample_space_edges.append(predecs[i] + [-1])
    #its_total = np.prod([len(s) for s in sample_space_edges]) # potential overflow!
    log_its_total = np.sum([np.log(len(s)) for s in sample_space_edges]) 

    if fix_size:
        if n_calc_nodes > 0:
            lengths_prev = []
            for i in range(2*(len(inter_nodes) - 1)):
                lengths_prev.append(len(sample_space_edges[i]))
            for i in range(2*(len(outp_nodes))):
                lengths_prev.append(len(sample_space_edges[i + 2*len(inter_nodes)]) - 1)
            log_its_prev = np.sum([np.log(l) for l in lengths_prev])
            log_its_total = logsumexp([log_its_total, log_its_prev], b = [1, -1])

    if log_its_total > np.log(max_orders):
        # just sample random orders
        possible_edges = []
        for _ in range(max_orders):
            order = []
            for tmp in sample_space_edges:
                order.append(np.random.choice(tmp))
            possible_edges.append(order)
    else:
        possible_edges = itertools.product(*sample_space_edges)
    valid_set = set()
    build_orders = []

    if verbose == 2:
        total_its = np.prod([len(s) for s in sample_space_edges])
        pbar = tqdm(possible_edges, total = total_its)
    else: 
        pbar = possible_edges

    for order in pbar:
        # order -> ID
        valid = True
        for i in range(l + n):
            if order[2*i] < order[2*i + 1]:
                valid = False
                break
        if valid:
            order_ID = ()
            # pre orders of outputs
            for i in range(n):
                order_ID = order_ID + get_pre_order(order, m + k + i, inp_nodes, inter_nodes, outp_nodes)
            
            is_new = True
            if fix_size:
                if n_calc_nodes > 0:
                    # how many intermediate nodes occur?
                    check_set = set(order_ID)
                    is_new = np.all([i in check_set for i in inter_nodes])

            if is_new:
                # rename intermediate nodes after the order in which they appear ( = 1 naming)
                ren_dict = {}
                counter = 0
                for i in order:
                    if (i in inter_nodes) and (i not in ren_dict):
                        ren_dict[i] = inter_nodes[counter]
                        counter += 1
                tmp_ID = tuple([ren_dict[node] if node in ren_dict else node for node in order_ID])
                is_new = tmp_ID not in valid_set
            
        
            
            if is_new:
                valid_set.add(tmp_ID)
                
                # build (node, parents) order for intermediate + output nodes
                tmp = sorted([i for i in set(order_ID) if i in inter_nodes])
                ren_dict = {node : inter_nodes[i] for i, node in enumerate(tmp)}
                build_order = []
                for i in sorted(tmp + outp_nodes):
                    if i in ren_dict:
                        # intermediate node
                        i1 = 2*(i - (m + k + n))
                        i2 = 2*(i - (m + k + n)) + 1
                    elif i in outp_nodes:
                        # output node
                        i1 = 2*l + 2*(i - (m + k))
                        i2 = 2*l + 2*(i - (m + k)) + 1
                    p1 = order[i1]
                    p2 = order[i2]
                    preds = []
                    if p1 in ren_dict:
                        preds.append(ren_dict[p1])
                    else:
                        preds.append(p1)
                    if p2 in ren_dict:
                        preds.append(ren_dict[p2])
                    elif p2 >= 0:
                        preds.append(p2)

                    if i in ren_dict:
                        build_order.append((ren_dict[i], tuple(preds)))
                    else:
                        build_order.append((i, tuple(preds)))
                build_orders.append(tuple(build_order))
                
    return build_orders

def evaluate_cgraph(cgraph:comp_graph.CompGraph, X:np.ndarray, loss_fkt:callable, opt_mode:str = 'grid_zoom', loss_thresh:float = None) -> tuple:
    '''
    Dummy function. Optimizes for constants.

    @Params:
        cgraph... computational DAG with constant input nodes
        X... input for DAG
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        opt_mode... one of {pool, opt, grid, grid_opt, grid_zoom}
        loss_thresh... only set in multiprocessing context - to communicate between processes

    @Returns:
        tuple of consts = array of optimized constants, loss = float of loss
    '''
    evaluate = True
    if loss_thresh is not None:
        # we are in parallel mode
        global stop_var
        evaluate = not bool(stop_var)


    if (not loss_fkt.opt_const) or (cgraph.n_consts == 0):
        return np.array([]), loss_fkt(X, cgraph, np.array([]))

    if evaluate:

        assert opt_mode in ['pool', 'opt', 'grid', 'grid_opt', 'grid_zoom'], 'Mode has to be one of {pool, opt, grid, grid_opt}'

        if opt_mode == 'pool':
            consts, loss = get_consts_pool(cgraph, X, loss_fkt)
        elif opt_mode == 'opt':
            consts, loss = get_consts_opt(cgraph, X, loss_fkt)
        elif opt_mode == 'grid':
            consts, loss = get_consts_grid(cgraph, X, loss_fkt)
        elif opt_mode == 'grid_zoom':
            consts, loss = get_consts_grid_zoom(cgraph, X, loss_fkt)
        elif opt_mode == 'grid_opt':
            consts, loss = get_consts_grid(cgraph, X, loss_fkt)
            consts, loss = get_consts_opt(cgraph, X, loss_fkt, c_start=consts)

        if loss_thresh is not None:
            if loss <= loss_thresh:
                stop_var = True
        return consts, loss
    else:
        return np.array([]), np.inf
        
def evaluate_build_order(order:list, m:int, n:int, k:int, X:np.ndarray, loss_fkt:callable, opt_mode:str = 'grid', loss_thresh:float = None) -> tuple:
    '''
    Given a build order (output of get_build_orders), tests all possible assignments of operators.

    @Params:
        order... list of tuples (node, parent_nodes)
        m... number of input nodes
        n... number of output nodes
        k... number of constant nodes
        X... input for DAGs
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        opt_mode... one of {pool, opt, grid, grid_opt}
        loss_thresh... only set in multiprocessing context - to communicate between processes
        
    @Returns:
        tuple:
            constants... list of optimized constants
            losses... list of losses for DAGs
            ops... list of ops that were tried
    '''

    
    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

    outp_nodes = [m + k + i for i in range(n)]
    op_spaces = []

    transl_dict = {}
    for i, (node, parents) in enumerate(order):
        if len(parents) == 2:
            op_spaces.append(bin_ops)
        else:
            if node in outp_nodes:
                op_spaces.append(un_ops)
            else:
                op_spaces.append([op for op in un_ops if op != '='])
        transl_dict[node] = i            

    ret_consts = []
    ret_losses = []
    ret_ops = []

    cgraph = None

    inv_array = []
    inv_mask = []
    for ops in itertools.product(*op_spaces):

        if len(inv_array) > 0:
            num_ops = np.array([config.NODE_ID[op] for op in ops])
            is_inv = np.sum((inv_array - num_ops)*inv_mask, axis = 1)
            is_inv = np.any(is_inv == 0)
        else:
            is_inv = False

        if not is_inv:
            if cgraph is None:
                cgraph = build_dag(order, ops, m, n, k)
            else:
                cgraph = adapt_ops(cgraph, order, ops)

            consts, loss = evaluate_cgraph(cgraph, X, loss_fkt, opt_mode, loss_thresh)

            if loss >= 1000 or (not np.isfinite(loss)):
                evaluate = True
                if loss_thresh is not None:
                    # we are in parallel mode
                    global stop_var
                    evaluate = not bool(stop_var)
                if evaluate:
                    # check for nonfinites
                    nonfins = cgraph.get_invalids(X, consts)
                    if (len(nonfins) < len(op_spaces)) and (len(nonfins) > 0):
                        tmp = np.zeros(len(op_spaces))
                        for node_idx in nonfins:
                            idx = transl_dict[node_idx]
                            tmp[idx] = config.NODE_ID[ops[idx]]
                        if len(inv_array) == 0:
                            inv_array = tmp.reshape(1, -1)
                        else:
                            inv_array = np.row_stack([inv_array, tmp])
                        inv_mask = (inv_array > 0).astype(int)
                


            ret_consts.append(consts)
            ret_losses.append(loss)
            ret_ops.append(ops)
        else:
            pass
    return ret_consts, ret_losses, ret_ops

def evaluate_build_order_old(order:list, m:int, n:int, k:int, X:np.ndarray, loss_fkt:callable, opt_mode:str = 'grid', loss_thresh:float = None) -> tuple:
    '''
    Given a build order (output of get_build_orders), tests all possible assignments of operators.

    @Params:
        order... list of tuples (node, parent_nodes)
        m... number of input nodes
        n... number of output nodes
        k... number of constant nodes
        X... input for DAGs
        loss_fkt... function f where f(X, graph, const) indicates how good the DAG is
        opt_mode... one of {pool, opt, grid, grid_opt}
        loss_thresh... only set in multiprocessing context - to communicate between processes
        
    @Returns:
        tuple:
            constants... list of optimized constants
            losses... list of losses for DAGs
            ops... list of ops that were tried
    '''

    
    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

    outp_nodes = [m + k + i for i in range(n)]
    op_spaces = []

    for i, (node, parents) in enumerate(order):
        if len(parents) == 2:
            op_spaces.append(bin_ops)
        else:
            if node in outp_nodes:
                op_spaces.append(un_ops)
            else:
                op_spaces.append([op for op in un_ops if op != '='])       

    ret_consts = []
    ret_losses = []
    ret_ops = []

    cgraph = None

    for ops in itertools.product(*op_spaces):

        
        if cgraph is None:
            cgraph = build_dag(order, ops, m, n, k)
        else:
            cgraph = adapt_ops(cgraph, order, ops)

        consts, loss = evaluate_cgraph(cgraph, X, loss_fkt, opt_mode, loss_thresh)
   
        ret_consts.append(consts)
        ret_losses.append(loss)
        ret_ops.append(ops)

    return ret_consts, ret_losses, ret_ops


def sample_graph(m:int, n:int, k:int, n_calc_nodes:int) -> comp_graph.CompGraph:
    '''
    Samples a computational DAG.

    @Params:
        m... number of input nodes
        n... number of output nodes
        k... number of constant nodes
        n_calc_nodes... number of intermediate nodes

    @Returns:
        computational DAG
    '''


    # 1. Sample build order
    l = n_calc_nodes
    inp_nodes = [i for i in range(m + k)]
    outp_nodes = [i + m + k for i in range(n)]
    inter_nodes = [i + m + k + n for i in range(l)]

    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]

    predecs = {}
    for i in inp_nodes:
        predecs[i] = []

    for i in outp_nodes:
        predecs[i] = inp_nodes + inter_nodes
        
    for i in inter_nodes:
        predecs[i] = inp_nodes + [j for j in inter_nodes if j < i]

    # sample order from predecessors
    order = []
    for i in inter_nodes + outp_nodes:
        l1 = predecs[i]
        order.append(np.random.choice(l1))

        l2 = [j for j in (predecs[i] + [-1]) if j < order[-1]]
        order.append(np.random.choice(l2))

    # order -> ID
    dep_entries = [l*2 + 2*i for i in range(n)] + [l*2 + 2*i + 1 for i in range(n)] # order indicies that are dependend
    tmp = set([order[i] for i in dep_entries if order[i] not in inp_nodes and order[i] >= 0]) # node indices that remain
    while len(tmp) > 0:
        tmp_deps = []
        for idx in tmp:
            tmp_deps.append(2*(idx - (m + k + n)))
            tmp_deps.append(2*(idx - (m + k + n)) + 1)
        tmp_deps = list(set(tmp_deps))
        dep_entries = dep_entries + tmp_deps
        tmp = set([order[i] for i in tmp_deps if order[i] not in inp_nodes and order[i] >= 0])
    dep_entries = sorted(dep_entries)
        
    ren_dict = {}
    order_ID = []
    counter = m + k
    for i in range(m + k, m + k + n + l):
        if i in outp_nodes:
            i1 = 2*l + 2*(i - (m + k))
            i2 = 2*l + 2*(i - (m + k)) + 1
        else:
            i1 = 2*(i - (m + k + n))
            i2 = 2*(i - (m + k + n)) + 1
        preds = []
        if i1 in dep_entries:
            preds.append(order[i1])
        if i2 in dep_entries and order[i2] >= 0:
            preds.append(order[i2])

        if len(preds) > 0:
            ren_dict[i] = counter
            order_ID.append((i, tuple(preds)))
            counter += 1
    new_order_ID = []
    for i, preds in order_ID:
        new_preds = [j if j not in ren_dict else ren_dict[j] for j in preds]
        new_order_ID.append((ren_dict[i], tuple(new_preds)))
    order = tuple(new_order_ID)
        
    # 2. sample operations on build order
    node_ops = []
    for _, parents in order:
        if len(parents) == 2:
            node_ops.append(np.random.choice(bin_ops))
        else:
            node_ops.append(np.random.choice(un_ops))

    # 3. create cgraph
    return build_dag(order, node_ops, m, n, k)

########################
# Search Methods
########################

def init_process(early_stop):
    global stop_var
    stop_var = early_stop 

def is_pickleable(x:object) -> bool:
    '''
    Used for multiprocessing. Loss function must be pickleable.

    @Params:
        x... an object

    @Returns:
        True if object can be pickled, False otherwise
    '''

    try:
        pickle.dumps(x)
        return True
    except (pickle.PicklingError, AttributeError):
        return False

def exhaustive_search(X:np.ndarray, n_outps: int, loss_fkt: callable, k: int, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, opt_mode:str = 'grid', max_orders:int = 10000, stop_thresh:float = -1.0, unique_loss:bool = True, **params) -> dict:
    '''
    Exhaustive search for a DAG.

    @Params:
        X... input for DAG, something that is accepted by loss fkt
        n_outps... number of outputs for DAG
        loss_fkt... function: X, cgraph, consts -> float
        k... number of constants
        n_calc_nodes... how many intermediate nodes at most?
        n_processes... number of processes for evaluation
        topk... we return top k found graphs
        verbose... print modus 0 = no print, 1 = status messages, 2 = progress bars
        opt_mode... method for optimizing constants, one of {pool, opt, grid, grid_opt}
        max_orders... will at most evaluate this many chosen orders
        max_size... will only return at most this many graphs (sorted by loss)
        stop_thresh... if loss is lower than this, will stop evaluation (only for single process)
        unique_loss... only take graph into topk if it has a totally new loss
    @Returns:
        dictionary with:
            graphs -> list of computational DAGs
            consts -> list of constants
            losses -> list of losses

    '''

    n_processes = max(min(n_processes, multiprocessing.cpu_count()), 1)
    ctx = multiprocessing.get_context('spawn')

    if n_processes > 1:
        error_msg = 'Loss function must be serializable with pickle for > 1 processes.\n'
        error_msg += 'See dag_search.MSE_loss_fkt for an example.\n'
        error_msg += 'If this worked before, consider reloading your loss funktion.'
        assert is_pickleable(loss_fkt), error_msg

    m = X.shape[1]
    n = n_outps

    # collect computational graphs (no operations on nodes yet)
    orders = get_build_orders(m, n, k, n_calc_nodes, max_orders = max_orders, verbose=verbose)

    if verbose > 0:
        print(f'Total orders: {len(orders)}')
        print('Evaluating orders')


    top_losses = []
    top_consts = []
    top_ops = []
    top_orders = []
    loss_thresh = np.inf

    early_stop = False
    if n_processes == 1:
        # sequential
        losses = []
        if verbose == 2:
            pbar = tqdm(orders)
        else:
            pbar = orders
        for order in pbar:
            consts, losses, ops = evaluate_build_order(order, m, n, k, X, loss_fkt, opt_mode = opt_mode)
            for c, loss, op in zip(consts, losses, ops):
                
                if loss <= loss_thresh:
                    if unique_loss:
                        valid = loss not in top_losses
                    else:
                        valid = True

                    if valid:
                        if len(top_losses) >= topk:
                            repl_idx = np.argmax(top_losses)
                            top_consts[repl_idx] = c
                            top_losses[repl_idx] = loss
                            top_ops[repl_idx] = op
                            top_orders[repl_idx] = order
                        else:
                            top_consts.append(c)
                            top_losses.append(loss)
                            top_ops.append(op)
                            top_orders.append(order)
                        
                        loss_thresh = np.max(top_losses)
                        if verbose == 2:
                            pbar.set_postfix({'best_loss' : np.min(top_losses)})
                if loss < stop_thresh:
                    early_stop = True
                    break
            if early_stop:
                break
    else:
        args = [[order, m, n, k, X, loss_fkt, opt_mode, stop_thresh] for order in orders]
        if verbose == 2:
            pbar = tqdm(args, total = len(args))
        else:
            pbar = args

        with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
            pool_results = pool.starmap(evaluate_build_order, pbar)
        for i, (consts, losses, ops) in enumerate(pool_results):
            for c, loss, op in zip(consts, losses, ops):
                if loss <= loss_thresh:
                    if unique_loss:
                        valid = loss not in top_losses
                    else:
                        valid = True

                    if valid:
                        if len(top_losses) >= topk:
                            repl_idx = np.argmax(top_losses)
                            top_consts[repl_idx] = c
                            top_losses[repl_idx] = loss
                            top_ops[repl_idx] = op
                            top_orders[repl_idx] = orders[i]
                        else:
                            top_consts.append(c)
                            top_losses.append(loss)
                            top_ops.append(op)
                            top_orders.append(orders[i])
                        
                        loss_thresh = np.max(top_losses)
                        if verbose == 2:
                            pbar.set_postfix({'best_loss' : np.min(top_losses)})

    sort_idx = np.argsort(top_losses)
    top_losses = [top_losses[i] for i in sort_idx]
    top_consts = [top_consts[i] for i in sort_idx]
    top_orders = [top_orders[i] for i in sort_idx]
    top_ops = [top_ops[i] for i in sort_idx]
    top_graphs = []
    for order, ops in zip(top_orders, top_ops):
        cgraph = build_dag(order, ops, m, n, k)
        top_graphs.append(cgraph.copy())

    ret = {
        'graphs' : top_graphs,
        'consts' : top_consts,
        'losses' : top_losses}

    return ret

def sample_search(X:np.ndarray, n_outps: int, loss_fkt: callable, k: int, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, opt_mode:str = 'grid', n_samples:int = int(1e4), stop_thresh:float = -1.0, unique_loss:bool = True, **params) -> dict:
    '''
    Sampling search for a DAG.

    @Params:
        X... input for DAG, something that is accepted by loss fkt
        n_outps... number of outputs for DAG
        loss_fkt... function: X, cgraph, consts -> float
        k... number of constants
        n_calc_nodes... how many intermediate nodes at most?
        n_processes... number of processes for evaluation
        topk... we return top k found graphs
        verbose... print modus 0 = no print, 1 = status messages, 2 = progress bars
        opt_mode... method for optimizing constants, one of {pool, opt, grid, grid_opt}
        n_samples... number of random graphs to check
        stop_thresh... if loss is lower than this, will stop evaluation (only for single process)
        unique_loss... only take graph into topk if it has a totally new loss
    @Returns:
        dictionary with:
            graphs -> list of computational DAGs
            consts -> list of constants
            losses -> list of losses
    '''
    n_processes = max(min(n_processes, multiprocessing.cpu_count()), 1)
    ctx = multiprocessing.get_context('spawn')

    if n_processes > 1:
        error_msg = 'Loss function must be serializable with pickle for > 1 processes.\n'
        error_msg += 'See dag_search.MSE_loss_fkt for an example.\n'
        error_msg += 'If this worked before, consider reloading your loss funktion.'
        assert is_pickleable(loss_fkt), error_msg

    m = X.shape[1]
    n = n_outps

    if verbose > 0:
        print('Generating graphs')
    if verbose == 2:
        pbar = tqdm(range(n_samples))
    else:
        pbar = range(n_samples)

    cgraphs = []
    for _ in pbar:
        cgraph = sample_graph(m, n, k, n_calc_nodes)
        cgraphs.append(cgraph.copy())

    if verbose > 0:
        print('Evaluating graphs')

    top_losses = []
    top_consts = []
    top_graphs = []
    loss_thresh = np.inf

    if n_processes == 1:
        # sequential
        if verbose == 2:
            pbar = tqdm(cgraphs)
        else:
            pbar = cgraphs
        for cgraph in pbar:
            c, loss = evaluate_cgraph(cgraph, X, loss_fkt, opt_mode)
            if loss <= loss_thresh:
                if unique_loss:
                    valid = loss not in top_losses
                else:
                    valid = True


                if valid:
                    if len(top_losses) >= topk:
                        repl_idx = np.argmax(top_losses)
                        top_consts[repl_idx] = c
                        top_losses[repl_idx] = loss
                        top_graphs[repl_idx] = cgraph.copy()
                    else:
                        top_consts.append(c)
                        top_losses.append(loss)
                        top_graphs.append(cgraph.copy())
                    
                    loss_thresh = np.max(top_losses)
                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses)})

            if loss <= stop_thresh:
                break
    else:

        early_stop = False
        args = [[cgraph, X, loss_fkt, opt_mode, stop_thresh] for cgraph in cgraphs]
        if verbose == 2:
            pbar = tqdm(args, total = len(args))
        else:
            pbar = args
        with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
            pool_results = pool.starmap(evaluate_cgraph, pbar)


        for i, (c, loss) in enumerate(pool_results):
            if loss <= loss_thresh:
                if unique_loss:
                    valid = loss not in top_losses
                else:
                    valid = True

                if valid:
                    if len(top_losses) >= topk:
                        repl_idx = np.argmax(top_losses)
                        top_consts[repl_idx] = c
                        top_losses[repl_idx] = loss
                        top_graphs[repl_idx] = cgraphs[i].copy()
                    else:
                        top_consts.append(c)
                        top_losses.append(loss)
                        top_graphs.append(cgraphs[i].copy())
                    
                    loss_thresh = np.max(top_losses)

    sort_idx = np.argsort(top_losses)
    top_graphs = [top_graphs[i] for i in sort_idx]
    top_consts = [top_consts[i] for i in sort_idx]
    top_losses = [top_losses[i] for i in sort_idx]
    
    ret = {
        'graphs' : top_graphs,
        'consts' : top_consts,
        'losses' : top_losses}

    return ret

def hierarchical_search(X:np.ndarray, n_outps: int, loss_fkt: callable, k: int, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, opt_mode:str = 'grid', max_orders:int = 10000, stop_thresh:float = -1.0, hierarchy_stop_thresh:float = -1.0, unique_loss:bool = True, **params) -> dict:
    '''
    Exhaustive search for a DAG but hierarchical.

    @Params:
        X... input for DAG, something that is accepted by loss fkt
        n_outps... number of outputs for DAG
        loss_fkt... function: X, cgraph, consts -> float
        k... number of constants
        n_calc_nodes... how many intermediate nodes at most?
        n_processes... number of processes for evaluation
        topk... we return top k found graphs
        verbose... print modus 0 = no print, 1 = status messages, 2 = progress bars
        opt_mode... method for optimizing constants, one of {pool, opt, grid, grid_opt}
        max_orders... will at most evaluate this many chosen orders
        max_size... will only return at most this many graphs (sorted by loss)
        stop_thresh... if loss is lower than this, will stop evaluation (only for single process)
        hierarchy_stop_thresh... if loss of any top performer is lower than this, will stop after hierarchy step
        unique_loss... only take graph into topk if it has a totally new loss
    @Returns:
        dictionary with:
            graphs -> list of computational DAGs
            consts -> list of constants
            losses -> list of losses

    '''

    n_processes = max(min(n_processes, multiprocessing.cpu_count()), 1)
    ctx = multiprocessing.get_context('spawn')

    if n_processes > 1:
        error_msg = 'Loss function must be serializable with pickle for > 1 processes.\n'
        error_msg += 'See dag_search.MSE_loss_fkt for an example.\n'
        error_msg += 'If this worked before, consider reloading your loss funktion.'
        assert is_pickleable(loss_fkt), error_msg

    m = X.shape[1]
    n = n_outps

    top_losses = []
    top_consts = []
    top_ops = []
    top_orders = []
    loss_thresh = np.inf

    for calc_nodes in range(n_calc_nodes + 1): # 0, 1, ..., n_calc_nodes
        if verbose > 0:
            print('#########################')
            print(f'# Calc Nodes: {calc_nodes}')
            print('#########################')

        # collect computational graphs (no operations on nodes yet)
        orders = get_build_orders(m, n, k, calc_nodes, max_orders = max_orders, verbose=verbose, fix_size=True)

        if verbose > 0:
            print(f'Total orders: {len(orders)}')
            print('Evaluating orders')

        early_stop = False
        if n_processes == 1:
            # sequential
            losses = []
            if verbose == 2:
                pbar = tqdm(orders)
            else:
                pbar = orders
            for order in pbar:
                consts, losses, ops = evaluate_build_order(order, m, n, k, X, loss_fkt, opt_mode = opt_mode)
                for c, loss, op in zip(consts, losses, ops):
                    
                    if loss <= loss_thresh:
                        if unique_loss:
                            valid = loss not in top_losses
                        else:
                            valid = True

                        if valid:
                            if len(top_losses) >= topk:
                                repl_idx = np.argmax(top_losses)
                                top_consts[repl_idx] = c
                                top_losses[repl_idx] = loss
                                top_ops[repl_idx] = op
                                top_orders[repl_idx] = order
                            else:
                                top_consts.append(c)
                                top_losses.append(loss)
                                top_ops.append(op)
                                top_orders.append(order)
                            
                            loss_thresh = np.max(top_losses)
                            if verbose == 2:
                                pbar.set_postfix({'best_loss' : np.min(top_losses)})
                    if loss < stop_thresh:
                        early_stop = True
                        break
                if early_stop:
                    break
        else:
            args = [[order, m, n, k, X, loss_fkt, opt_mode, stop_thresh] for order in orders]
            if verbose == 2:
                pbar = tqdm(args, total = len(args))
            else:
                pbar = args

            with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
                pool_results = pool.starmap(evaluate_build_order, pbar)

            for i, (consts, losses, ops) in enumerate(pool_results):
                for c, loss, op in zip(consts, losses, ops):
                    if loss <= loss_thresh:
                        if unique_loss:
                            valid = loss not in top_losses
                        else:
                            valid = True

                        if valid:
                            if len(top_losses) >= topk:
                                repl_idx = np.argmax(top_losses)
                                top_consts[repl_idx] = c
                                top_losses[repl_idx] = loss
                                top_ops[repl_idx] = op
                                top_orders[repl_idx] = orders[i]
                            else:
                                top_consts.append(c)
                                top_losses.append(loss)
                                top_ops.append(op)
                                top_orders.append(orders[i])
                            
                            loss_thresh = np.max(top_losses)
                            if verbose == 2:
                                pbar.set_postfix({'best_loss' : np.min(top_losses)})
                

        sort_idx = np.argsort(top_losses)
        top_losses = [top_losses[i] for i in sort_idx]
        top_consts = [top_consts[i] for i in sort_idx]
        top_orders = [top_orders[i] for i in sort_idx]
        top_ops = [top_ops[i] for i in sort_idx]
        top_graphs = []
        for order, ops in zip(top_orders, top_ops):
            cgraph = build_dag(order, ops, m, n, k)
            top_graphs.append(cgraph.copy())

        if top_losses[0] <= stop_thresh or np.any(np.array(top_losses) <= hierarchy_stop_thresh):
            if verbose > 0:
                print(f'Stopping because early stop criteria has been matched!')
            break

    ret = {
        'graphs' : top_graphs,
        'consts' : top_consts,
        'losses' : top_losses}

    return ret



########################
# Sklearn Interface for Symbolic Regression Task
########################


class DAGRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for exhaustive search.
    '''

    def __init__(self, k:int = 1, n_calc_nodes:int = 5, max_orders:int = int(1e5), random_state:int = None, processes:int = 1, max_samples:int = 100, mode : str = 'exhaustive', loss_fkt :DAG_Loss_fkt = MSE_loss_fkt, **kwargs):
        '''
        @Params:
            k.... number of constants
            n_calc_nodes... number of possible intermediate nodes
            max_orders... maximum number of expression - skeletons in search
            random_state... for reproducibility
            processes... number of processes for multiprocessing
            max_samples... maximum number of samples on which to fit
            mode... one of 'exhaustive' or 'hierarchical'
            loss_fkt... loss function class
        '''
        self.k = k
        self.n_calc_nodes = n_calc_nodes
        self.max_orders = max_orders
        self.max_samples = max_samples
        assert mode in ['exhaustive', 'hierarchical'], f'Search mode {mode} is not supported.'
        self.mode = mode

        self.processes = max(min(processes, multiprocessing.cpu_count()), 1)

        self.random_state = random_state
        self.loss_fkt = loss_fkt

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 1):
        '''
        Fits a model on given regression data.
        @Params:
            X... input data (shape n_samples x inp_dim)
            y... output data (shape n_samples)
            processes... number of processes for evaluation
        '''
        assert len(y.shape) == 1, f'y must be 1-dimensional (current shape: {y.shape})'

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if len(X) > self.max_samples:
            sub_idx = np.arange(len(X))
            np.random.shuffle(sub_idx)
            sub_idx = sub_idx[:self.max_samples]
            X_sub = X[sub_idx]
            y_sub = y[sub_idx]
        else:
            X_sub = X
            y_sub = y

        y_part = y_sub.reshape(-1, 1)
        m = X_sub.shape[1]
        n = 1
        loss_fkt = self.loss_fkt(y_part)
        params = {
            'X' : X_sub,
            'n_outps' : n,
            'loss_fkt' : loss_fkt,
            'k' : self.k,
            'n_calc_nodes' : self.n_calc_nodes,
            'n_processes' : self.processes,
            'topk' : 10,
            'opt_mode' : 'grid_zoom',
            'verbose' : verbose,
            'max_orders' : self.max_orders, 
            'stop_thresh' : 1e-10
        }
        if self.mode == 'hierarchical':
            res = hierarchical_search(**params)
        else:
            res = exhaustive_search(**params)

        # optimizing constants of top DAGs
        if verbose > 0:
            print('Optimizing best constants')
        loss_fkt = self.loss_fkt(y.reshape(-1, 1))
        losses = []
        consts = []
        for graph, c in zip(res['graphs'], res['consts']):
            new_c, loss = get_consts_opt(graph, X, loss_fkt, c_start = c)
            losses.append(loss)
            consts.append(new_c)
        best_idx = np.argmin(losses)
        if verbose > 0:
            print(f'Found graph with loss {losses[best_idx]}')

        
        self.cgraph = res['graphs'][best_idx]
        self.consts = consts[best_idx]
        return self

    def predict(self, X:np.ndarray, return_grad : bool = False):
        '''
        Predicts values for given samples.

        @Params:
            X... input data (shape n_samples x inp_dim)
            return_grad... whether to return gradient wrt. input at X

        @Returns:
            predictions (shape n_samples)
            [if wanted: gradient (shape n_samples x inp_dim)]
        '''
        assert hasattr(self, 'cgraph'), 'No graph found yet. Call .fit first!'
        if return_grad:
            pred, grad = self.cgraph.evaluate(X, c = self.consts, return_grad = return_grad)
            return pred[:, 0], grad[0]

        else:
            pred = self.cgraph.evaluate(X, c = self.consts, return_grad = return_grad)
            return pred[:, 0]

    def model(self):
        '''
        Evaluates symbolic expression.
        '''
        assert hasattr(self, 'cgraph'), 'No graph found yet. Call .fit first!'
        exprs = self.cgraph.evaluate_symbolic(c = self.consts)
        return exprs[0]

    def complexity(self):
        '''
        Complexity of expression (number of calculations)
        '''
        assert hasattr(self, 'cgraph'), 'No graph found yet. Call .fit first!'
        return self.cgraph.n_operations()
    
########################
# New: Substitution Regressor
# ########################


def find_substitutions(X:np.ndarray, y:np.ndarray, regr_bb, n_calc_nodes:int = 2, verbose:int = 2, hierarchy_stop_thresh:float = 1e-2, n_processes:int = 1, mode:str = 'gradient', topk:int = 10) -> comp_graph.CompGraph:
    # 1. find best substitution on subset

    if mode == 'gradient':
        loss_fkt_simpl = Gradient_loss_fkt(regr_bb, X, y)
    else:
        loss_fkt_simpl = Fit_loss_fkt(regr_bb, y)

    params = {
        'X' : X,
        'n_outps' : 1,
        'loss_fkt' : loss_fkt_simpl,
        'k' : 0,
        'n_calc_nodes' : n_calc_nodes,
        'n_processes' : n_processes,
        'topk' : topk,
        'verbose' : verbose,
        'max_orders' : 10000, 
        'hierarchy_stop_thresh' : hierarchy_stop_thresh
    }
    res = hierarchical_search(**params)

    return res['graphs'], res['losses']

def find_best_substitutions(X:np.ndarray, y:np.ndarray, regr_bb, verbose:int = 2, beamsize:int = 5, topk:int = 3, n_calc_nodes:int = 2, mode:str = 'gradient', hierarchy_stop_thresh:float = 1e-3, n_processes:int = 1, random_state:int = 0):
    np.random.seed(random_state)
    # beam consists of tuples (data, translation)

    var_dict = {f'x_{i}' : f'z_{i}' for i in range(X.shape[1])}
    orig_tuple = (X, var_dict)

    final_beam = [orig_tuple]
    final_losses = [0.0]
    unique_dicts = set()

    current_beam = [orig_tuple]
    new_beam = []
    done = False



    while not done:
        if verbose > 0:
            print('##############')
            print(f'# Evaluating Beam of Size {len(current_beam)}')
            print('##############')
            
        
            
        for X_current, var_dict in current_beam:        
            if X_current.shape[1] > 1:
                if verbose > 0:
                    print(f'Searching for Simplification of')
                    for s in var_dict:
                        print(f'{s} -> {var_dict[s]}')
                
                
                graphs, losses = find_substitutions(X_current, y, regr_bb, n_calc_nodes = n_calc_nodes, verbose = verbose, mode = mode, topk = beamsize, n_processes=n_processes, hierarchy_stop_thresh=hierarchy_stop_thresh)

                for graph, loss in zip(graphs, losses):
                    
                    if (loss <= np.min(final_losses)) or (len(final_losses) < topk):

                        # carry out substitution
                        sub_expr = graph.evaluate_symbolic()[0]
                        sub_loss = loss
                        if verbose > 0:
                            print(f'Substitution: {sub_expr}\tLoss: {sub_loss}')

                        # substitute
                        used_idxs = sorted([int(str(e).split('_')[-1]) for e in sub_expr.free_symbols])
                        X_new = graph.evaluate(X_current, np.array([]))
                        X_new = np.column_stack([X_new] + [X_current[:, i] for i in range(X_current.shape[1]) if i not in used_idxs])

                        sub_expr = str(sub_expr)
                        for i in range(X_current.shape[1]-1 , -1, -1):
                            s = f'x_{i}'
                            sub_expr = sub_expr.replace(s, f'({var_dict[s]})')

                        new_var_dict = {f'x_0' : sub_expr}
                        for i in range(X_current.shape[1]):
                            if i not in used_idxs:
                                s_new = f'x_{len(new_var_dict)}'
                                s_old = f'x_{i}'
                                new_var_dict[s_new] = var_dict[s_old]

                        dict_tuple = tuple([new_var_dict[f'x_{i}'] for i in range(len(new_var_dict))])
                        if dict_tuple not in unique_dicts:
                            unique_dicts.add(dict_tuple)


                            # add to new beam
                            new_beam.append((X_new, new_var_dict))

                            # add to final solutions (+ update)
                            final_losses.append(loss)
                            final_beam.append((X_new, new_var_dict))
                            sort_idxs = np.argsort(final_losses)
                            final_losses = [final_losses[i] for i in sort_idxs[:topk]]
                            final_beam = [final_beam[i] for i in sort_idxs[:topk]]


                            
        
        done = (len(new_beam) == 0)
        current_beam = new_beam
        new_beam = []


    # sort by size
    sizes = [X_beam.shape[1] for X_beam, _ in final_beam]
    sort_idxs = np.argsort(sizes)
    final_beam = [final_beam[i] for i in sort_idxs]
    final_losses = [final_losses[i] for i in sort_idxs]

    return final_beam, final_losses

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

class SimplificationRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for symbolic Regressor based on simplification strategies.
    '''

    def __init__(self, random_state:int = None, regr_search = None, regr_blackbox = None, simpl_nodes:int = 2, processes:int = 1, mode = 'gradient'):
        self.random_state = random_state
        self.processes = processes
        self.regr_search = regr_search
        self.regr_blackbox = regr_blackbox
        self.simpl_nodes = simpl_nodes
        self.mode = mode

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 0):
        if self.regr_search is None:
            self.regr_search = DAGRegressor(processes=self.processes, random_state = self.random_state)
        if self.regr_blackbox is None:
            # select blackbox from test performance
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            polydegrees = np.arange(1, 5, 1)
            test_r2s = []
            for degree in polydegrees:
                regr_bb = PolyReg(degree = degree)
                regr_bb.fit(X_train, y_train)
                pred = regr_bb.predict(X_test)
                test_r2s.append(r2_score(y_test, pred))
            self.regr_blackbox = PolyReg(degree = polydegrees[np.argmax(test_r2s)])


        if verbose > 0:
            print('Searching for Simplifications')
        subs, _ = find_best_substitutions(X, y, self.regr_blackbox, verbose = verbose, n_processes=self.processes, n_calc_nodes=self.simpl_nodes, mode = self.mode)
        
        if verbose > 0:
            print('Searching for Expression')

        # testing the best substitutions
        r2s = []
        exprs = []
        for X_sub, var_dict in subs:
            if verbose > 0:
                print('Substitution:')
                for s in var_dict:
                    print(f'{s} -> {var_dict[s]}')

            self.regr_search.fit(X_sub, y, verbose = verbose)
            pred = self.regr_search.predict(X_sub)

            if np.all(np.isfinite(pred)):
                r2 = r2_score(y, pred)
            else:
                r2 = -np.inf
            r2s.append(r2)
            exprs.append(self.regr_search.model())
            if r2 > 1-1e-3:
                break
        best_idx = np.argmax(r2s)

        # save models
        self.exprs = exprs
        self.r2s = r2s

        expr = exprs[best_idx]
        X_simpl, var_dict = subs[best_idx]

        

        expr_str = str(expr)
        for i in range(X_simpl.shape[1]-1 , -1, -1):
            s = f'x_{i}'
            expr_str = expr_str.replace(s, f'({var_dict[s]})')
        expr_str = expr_str.replace('z_', 'x_')
        self.expr = sympy.sympify(expr_str)
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
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

class SimplificationRegressorOld(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for symbolic Regressor based on simplification strategies.
    '''

    def __init__(self, random_state:int = None, regr_search = None, regr_blackbox = None, simpl_nodes:int = 1, n_processes:int = 1, mode:str = 'gradient'):
        self.random_state = random_state
        if regr_search is None:
            regr_search = DAGRegressor(random_state = random_state)
        if regr_blackbox is None:
            regr_blackbox = PolyReg(degree = 5)
        self.regr_search = regr_search
        self.regr_blackbox = regr_blackbox
        self.simpl_nodes = simpl_nodes
        self.n_processes = n_processes
        assert mode in ['gradient', 'fit']
        self.mode = mode

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 1):
        
        
        var_dict = {f'x_{i}' : f'z_{i}' for i in range(X.shape[1])}
        done = False
        X_tmp = X.copy()
        

        self.exprs = []
        self.r2_scores = []
        
        
        while not done:
            
            if verbose > 0:
                print('Variable Dictionary')
                for s in var_dict:
                    print(f'{s} -> {var_dict[s]}')
            
            
                
            # 1. try to solve with symbolic regressor
            if verbose > 0:
                print('Trying to solve...')
            self.regr_search.fit(X_tmp, y, verbose = verbose)
            

            # 2. Evaluating solution
            pred = self.regr_search.predict(X_tmp)
            r2 = r2_score(y, pred)
            self.r2_scores.append(r2)
            
            # 3. Re-Substitute
            expr = str(self.regr_search.model())
            expr_raw = expr
            for i in range(X_tmp.shape[1]-1 , -1, -1):
                s = f'x_{i}'
                expr = expr.replace(s, f'({var_dict[s]})')
            expr = expr.replace('z_', 'x_')
            expr = sympy.sympify(expr)
            self.exprs.append(expr)
            
            if verbose > 0:
                print(f'Expression: {expr_raw}\tR2-Score: {r2}')
            
            # 4. done?
            #if len(r2_scores) > 1:
            #    done = r2_scores[-1] < r2_scores[-2]
            done = done or (r2 > (1-1e-5))
            done = done or (X_tmp.shape[1] == 1)
            
            if not done:
                # 5. Find Substitution
                if verbose > 0:
                    print('Not done, searching for Substitution...') 
                cgraphs, sub_r2s = find_substitutions(X_tmp, y, self.regr_blackbox, verbose = verbose, n_processes=self.n_processes, mode = self.mode)

                # 6. Substitute
                cgraph = cgraphs[0]
                sub_r2 = sub_r2s[0]
                sub_expr = cgraph.evaluate_symbolic()[0]

                
                used_idxs = sorted([int(str(e).split('_')[-1]) for e in sub_expr.free_symbols])
                X_new = cgraph.evaluate(X_tmp, np.array([]))
                X_new = np.column_stack([X_new] + [X_tmp[:, i] for i in range(X_tmp.shape[1]) if i not in used_idxs])
                
                sub_expr = str(sub_expr)
                sub_expr_raw = sub_expr
                for i in range(X_tmp.shape[1]-1 , -1, -1):
                    s = f'x_{i}'
                    sub_expr = sub_expr.replace(s, f'({var_dict[s]})')
                
                new_var_dict = {f'x_0' : sub_expr}
                for i in range(X_tmp.shape[1]):
                    if i not in used_idxs:
                        s_new = f'x_{len(new_var_dict)}'
                        s_old = f'x_{i}'
                        new_var_dict[s_new] = var_dict[s_old]
                X_tmp = X_new
                var_dict = new_var_dict
                
                sub_expr = sub_expr.replace('z_', 'x_')
                sub_expr = sympy.sympify(sub_expr)
                if verbose > 0:
                    print(f'Substitution: {sub_expr_raw}\tLoss: {sub_r2}')
                    
            elif verbose > 0:
                print('done.')

        self.expr = self.exprs[np.argmax(self.r2_scores)]
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
        self.exec_func = sympy.lambdify(x_symbs, self.expr)
         
        return self
    
    def predict(self, X):
        assert hasattr(self, 'exprs')
        assert len(self.exprs) > 0

        if not hasattr(self, 'exec_func'):
            self.expr = self.exprs[np.argmax(self.r2_scores)]
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
        assert hasattr(self, 'exprs'), 'No expressions found yet. Call .fit first!'
        assert len(self.exprs) > 0

        if not hasattr(self, 'expr'):
            self.expr = self.exprs[np.argmax(self.r2_scores)]

        return self.expr
    
########################
# New: Replacement Regressor
# ########################

class PolySubRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for symbolic Regressor based on replacement strategies.
    '''

    def __init__(self, random_state:int = None, regr_search = None, simpl_nodes:int = 3, topk:int = 3, max_orders:int = int(1e5), processes:int = 1):
        self.random_state = random_state
        self.processes = processes
        self.regr_search = regr_search
        self.regr_poly = None
        self.simpl_nodes = simpl_nodes
        self.max_orders = max_orders
        self.topk = topk

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 0):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.regr_search is None:
            self.regr_search = DAGRegressor(processes=self.processes, random_state = self.random_state)
        
        # fitting poly
        fit_thresh = 1-1e-3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        polydegrees = np.arange(1, 7, 1)
        test_r2s = []
        found = False
        for degree in polydegrees:
            regr_poly = PolyReg(degree = degree)
            regr_poly.fit(X_train, y_train)
            pred = regr_poly.predict(X_test)
            s = r2_score(y_test, pred)
            if s > fit_thresh:
                found = True
                break
            else:
                test_r2s.append(s)
        
        if found:
            self.regr_poly = regr_poly
        else:
            self.regr_poly = PolyReg(degree = polydegrees[np.argmax(test_r2s)])


        if verbose > 0:
            print('Searching for Replacements')

        loss_fkt = Repl_loss_fkt(self.regr_poly, y)
        params = {
            'X' : X,
            'n_outps' : 1,
            'loss_fkt' : loss_fkt,
            'k' : 0,
            'n_calc_nodes' : self.simpl_nodes,
            'n_processes' : self.processes,
            'topk' : self.topk,
            'opt_mode' : 'grid_zoom',
            'verbose' : verbose,
            'max_orders' : self.max_orders, 
            'stop_thresh' : 1e-20
        }
        res = exhaustive_search(**params)

        if (res['losses'][0] < 1e-20):
            # we solved it using a polynomial
            if verbose > 0:
                print('Solving with Polynomial')
            graph = res['graphs'][0]
            repl_expr = graph.evaluate_symbolic()[0]
            repl_idx = loss_fkt(X, graph, [], True)

            X_new = np.delete(X, repl_idx, axis = 1)
            X_new = np.column_stack([graph.evaluate(X, np.array([]))[:, 0], X_new])
            
            self.regr_poly.fit(X_new, y)
            expr = utils.round_floats(self.regr_poly.model())

            # translate back
            expr = self._translate(X, repl_idx, expr, repl_expr)
            
        else:
            # we couldnt solve it, try the best replacements + original problem
            if verbose > 0:
                print('Solving with Symbolic Regressor')
            scores = []
            exprs = []
            found = False
            for graph in res['graphs']:
                repl_expr = graph.evaluate_symbolic()[0]
                repl_idx = loss_fkt(X, graph, [], True)

                if verbose > 1:
                    print(f'Replacement: {repl_expr}\nIndices: {repl_idx}')
                
                X_new = np.delete(X, repl_idx, axis = 1)
                X_new = np.column_stack([graph.evaluate(X, np.array([]))[:, 0], X_new])

                self.regr_search.fit(X_new, y, verbose = verbose)

                expr = self.regr_search.model()
                exprs.append(expr)

                
                pred = self.regr_search.predict(X_new)
                score = r2_score(y, pred)
                scores.append(score)

                if score == 1.0:
                    # we found the solution
                    found = True
                    break

            best_idx = np.argmax(scores)
            repl_expr = res['graphs'][best_idx].evaluate_symbolic()[0]
            repl_idx = loss_fkt(X, graph, [], True)
            expr = exprs[best_idx]
            expr = self._translate(X, repl_idx, expr, repl_expr)


            if not found:
                # also try original problem
                
                if verbose > 1:
                    print(f'Original Problem')
                self.regr_search.fit(X, y, verbose = verbose)
                pred = self.regr_search.predict(X)
                score = r2_score(y, pred)
                if score > scores[best_idx]:
                    expr = self.regr_search.model()




        self.expr = expr
        x_symbs = [f'x_{i}' for i in range(X.shape[1])]
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
    
    def _translate(self, X, repl_idx, expr, repl_expr):
        '''
        Translates the expression back.

        @Params:
            X... original data
            X_new... transformed data
            repl_idx... indices that have been replaced
            expr... final expression for X_new
            repl_expr... expression for replacement

        @Returns:
            translated expression
        '''

        orig_idx = np.arange(X.shape[1])
        new_idx = np.concatenate([np.array([-1]), np.delete(orig_idx, repl_idx)])
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
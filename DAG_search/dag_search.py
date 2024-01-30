'''
Operations for combining computational graphs
'''
import numpy as np
import itertools
import warnings
import sympy
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import stats
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle
import multiprocessing
from copy import deepcopy
import time
import sklearn
import sys

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from DAG_search import config
from DAG_search import comp_graph
from DAG_search import utils


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

## Substitution

class Feature_loss_fkt(DAG_Loss_fkt):
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
        self.regr = regr
        self.y = y
        if hasattr(self.regr, 'fit'):
            self.regr.fit(X, y)
        self.df_dx = utils.est_gradient(self.regr, X, y)
        
        
    def __call__(self, X:np.ndarray, cgraph:comp_graph.CompGraph, c:np.ndarray) -> np.ndarray:
        '''
        Lossfkt(X, graph, consts)

        @Params:
            X... input for DAG (N x m)
            cgraph... computational Graph
            c... array of constants (2D)

        @Returns:
            Median absolute error of gradients for substituted variables if we use graph as substitution
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

            expr = cgraph.evaluate_symbolic()[0]
            I = sorted([int(str(e).split('_')[-1]) for e in expr.free_symbols if str(e).startswith('x_')])
            
            df_dxI = self.df_dx[:, I]
            df_dxI_norm = (df_dxI.T/np.linalg.norm(df_dxI, axis = 1)).T # shape N x I
            
            if len(I) > 1:

                h_x, dh_dx = cgraph.evaluate(X, c, return_grad = True) # shape r x 1 x N x inps

                valids = np.all(np.isfinite(h_x).reshape(r, -1), axis=-1)
                valids = valids & np.all((h_x < 10000).reshape(r, -1), axis=-1)

                dh_dx = dh_dx[:, 0, :, :] # shape r x N x inps
                dh_dxI = dh_dx[:, :, I] # shape r x N x I

                hI_norm = np.linalg.norm(dh_dxI, axis = -1) # shape r x N
                dh_dxI_norm = np.transpose(np.transpose(dh_dxI, (2, 0, 1))/hI_norm, (1, 2, 0)) # shape r x N x I
                #dh_dxI_norm = (dh_dxI - self.mu[I])/self.std[I]

                v1 = dh_dxI_norm.reshape(r, -1) # shape r x N*I
                v2 = df_dxI_norm.flatten() # shape N*I

                losses1 = np.median((v1 - v2)**2, axis = -1)
                losses2 = np.median((v1 + v2)**2, axis = -1)
                losses = np.minimum(losses1, losses2)

                losses[~valids] = np.inf

            else:
                losses = np.inf*np.ones(r)

            losses[~np.isfinite(losses)] = np.inf
            losses[np.isnan(losses)] = np.inf
        if not vec:
            return losses[0]
        else:
            return losses
        

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
        for tmp in sample_space_edges:
            possible_edges.append(np.random.choice(tmp, size = max_orders))
        possible_edges = np.column_stack(possible_edges)

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
        
def evaluate_build_order(order:list, m:int, n:int, k:int, X:np.ndarray, loss_fkt:callable, topk:int = 5, opt_mode:str = 'grid_zoom', loss_thresh:float = None, start_time:float = None, max_time:float = 3600, expect_evals:int = 10000, pareto:bool = False) -> tuple:
    '''
    Given a build order (output of get_build_orders), tests all possible assignments of operators.

    @Params:
        order...        list of tuples (node, parent_nodes)
        m...            number of input nodes
        n...            number of output nodes
        k...            number of constant nodes
        X...            input for DAGs
        loss_fkt...     function f where f(X, graph, const) indicates how good the DAG is
        topk...         number of top performers to be returned
        opt_mode...     one of {pool, opt, grid, grid_opt, grid_zoom} (see evaluate_cgraph function)
        loss_thresh...  only set in multiprocessing context - to communicate between processes for early stopping
        start_time...   if set, will not evaluate any orders after max_time seconds from start time
        max_time...     if set, will not evaluate any orders after max_time seconds from start time
        expect_evals... operator assignments are chosen so that we evaluate at most this many graphs in expectation
                        set to None if you want to go fully exhaustive
        pareto...       if set, will return the pareto front instead of the top k

    @Returns:
        tuple:
            constants... list of optimized constants
            losses... list of losses for DAGs
            ops... list of ops that were tried
    '''
    evaluate = True
    if loss_thresh is not None:
        # we are in parallel mode
        global stop_var
        evaluate = not bool(stop_var)
        
    ret_consts = []
    ret_losses = []
    ret_ops = []

    max_loss = np.inf
    max_idx = None

    if start_time is not None:
        evaluate = evaluate and (time.time() - start_time) < max_time

    if evaluate:
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

        if expect_evals is not None:
            v = np.log(expect_evals)
            log_space_size = 0
            for s in op_spaces:
                log_space_size += np.log(len(s))
            log_accept_ratio = min(0, v - log_space_size)
            accept_ratio = np.exp(log_accept_ratio)
        else:
            accept_ratio = 1.0


        cgraph = None

        # keeping track of invalids
        inv_array = []
        inv_mask = []
        for ops in itertools.product(*op_spaces):
            if (start_time is not None) and ((time.time() - start_time) >= max_time):
                break
            accept = True
            if accept_ratio < 1.0:
                accept = (np.random.rand() < accept_ratio)

            
            if len(inv_array) > 0:
                num_ops = np.array([config.NODE_ID[op] for op in ops])
                is_inv = np.sum((inv_array - num_ops)*inv_mask, axis = 1)
                is_inv = np.any(is_inv == 0)
            else:
                is_inv = False

            if not is_inv and accept:
                if cgraph is None:
                    cgraph = build_dag(order, ops, m, n, k)
                else:
                    cgraph = adapt_ops(cgraph, order, ops)

                consts, loss = evaluate_cgraph(cgraph, X, loss_fkt, opt_mode, loss_thresh)
                if loss >= 1000 or (not np.isfinite(loss)):
                    evaluate = True
                    if loss_thresh is not None:
                        # we are in parallel mode
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
                    
                if (len(ret_losses) == 0):
                    ret_consts.append(consts)
                    ret_losses.append(loss)
                    ret_ops.append(ops)
                    max_idx = np.argmax(ret_losses)
                    max_loss = ret_losses[max_idx]

                elif pareto:
                    # check which entries it dominates
                    dominated_entries = []
                    for i, (pareto_loss, pareto_ops) in enumerate(zip(ret_losses, ret_ops)):
                        if pareto_loss >= loss and len(pareto_ops) >= len(ops):
                            dominated_entries.append(i)
                    # keep only non dominated entries
                    if len(dominated_entries) > 0:
                        ret_consts = [consts] + [ret_consts[i] for i in range(len(ret_consts)) if i not in dominated_entries]
                        ret_losses = [loss] + [ret_losses[i] for i in range(len(ret_consts)) if i not in dominated_entries]
                        ret_ops = [ops] + [ret_ops[i] for i in range(len(ret_consts)) if i not in dominated_entries]
                    
                elif (len(ret_losses) < topk):
                    # append
                    ret_consts.append(consts)
                    ret_losses.append(loss)
                    ret_ops.append(ops)

                    max_idx = np.argmax(ret_losses)
                    max_loss = ret_losses[max_idx]
                else:
                    if loss < max_loss:
                        # replace
                        ret_consts[max_idx] = consts
                        ret_losses[max_idx] = loss
                        ret_ops[max_idx] = ops
                        
                        max_idx = np.argmax(ret_losses)
                        max_loss = ret_losses[max_idx]
            else:
                pass
    return ret_consts, ret_losses, ret_ops

def sample_graph(m:int, n:int, k:int, n_calc_nodes:int, only_order:bool = False) -> comp_graph.CompGraph:
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
        
    if only_order:
        return order
    
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

def exhaustive_search(X:np.ndarray, n_outps: int, loss_fkt: callable, k: int, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, opt_mode:str = 'grid', max_orders:int = 10000, max_time:float = 900.0, stop_thresh:float = -1.0, unique_loss:bool = True, pareto:bool = False, **params) -> dict:
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
        max_time... when this time (in seconds) has passed, no new orders are evaluated
        max_size... will only return at most this many graphs (sorted by loss)
        stop_thresh... if loss is lower than this, will stop evaluation (only for single process)
        unique_loss... only take graph into topk if it has a totally new loss
        pareto... if set, will return the pareto front instead of the top k
    @Returns:
        dictionary with:
            graphs -> list of computational DAGs
            consts -> list of constants
            losses -> list of losses

    '''
    ret = {
        'graphs' : [],
        'consts' : [],
        'losses' : []}
    process_start_time = time.time()

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
            consts, losses, ops = evaluate_build_order(order, m, n, k, X, loss_fkt, topk, opt_mode = opt_mode, start_time=process_start_time, max_time=max_time, pareto=pareto)
            
            if pareto:
                top_losses += losses
                top_consts += consts
                top_ops += ops
                top_orders += [order]*len(losses)
                
                complexities = [len(op) for op in top_ops]
                pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                if unique_loss:
                    pareto_losses = set()
                    take_idxs = []
                    for j in pareto_idxs:
                        v = np.round(top_losses[j], 15)
                        if v not in pareto_losses:
                            pareto_losses.add(v)
                            take_idxs.append(j)
                else:
                    take_idxs = pareto_idxs
                
                top_losses = [top_losses[i] for i in take_idxs]
                top_consts = [top_consts[i] for i in take_idxs]
                top_ops = [top_ops[i] for i in take_idxs]
                top_orders = [top_orders[i] for i in take_idxs]


                
                if verbose == 2:
                    pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})

                early_stop = np.any(np.array(top_losses) < stop_thresh)
            else:
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

        args = [[order, m, n, k, X, loss_fkt, topk, opt_mode, stop_thresh, process_start_time, max_time, pareto] for order in orders]
        if verbose == 2:
            pbar = tqdm(args, total = len(args))
        else:
            pbar = args

        with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
            pool_results = pool.starmap(evaluate_build_order, pbar)
        for i, (consts, losses, ops) in enumerate(pool_results):
            if early_stop:
                break
            if pareto:
                top_losses += losses
                top_consts += consts
                top_ops += ops
                top_orders += [orders[i]]*len(losses)
                
                complexities = [len(op) for op in top_ops]
                pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                if unique_loss:
                    pareto_losses = set()
                    take_idxs = []
                    for j in pareto_idxs:
                        v = np.round(top_losses[j], 15)
                        if v not in pareto_losses:
                            pareto_losses.add(v)
                            take_idxs.append(j)
                else:
                    take_idxs = pareto_idxs
                
                top_losses = [top_losses[j] for j in take_idxs]
                top_consts = [top_consts[j] for j in take_idxs]
                top_ops = [top_ops[j] for j in take_idxs]
                top_orders = [top_orders[j] for j in take_idxs]
                
                if verbose == 2:
                    pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})

                early_stop = np.any(np.array(top_losses) < stop_thresh)
            else:
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
                    if loss < stop_thresh:
                        early_stop = True
                        break

    sort_idx = np.argsort(top_losses)
    top_losses = [top_losses[i] for i in sort_idx]
    top_consts = [top_consts[i] for i in sort_idx]
    top_orders = [top_orders[i] for i in sort_idx]
    top_ops = [top_ops[i] for i in sort_idx]
    top_graphs = []
    for order, ops in zip(top_orders, top_ops):
        cgraph = build_dag(order, ops, m, n, k)
        top_graphs.append(cgraph.copy())

    ret['graphs'] = top_graphs
    ret['consts'] = top_consts
    ret['losses'] = top_losses

    return ret

def sample_search(X:np.ndarray, n_outps: int, loss_fkt: callable, k: int, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, opt_mode:str = 'grid', n_samples:int = int(1e4), stop_thresh:float = -1.0, unique_loss:bool = True, pareto:bool = False, **params) -> dict:
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
        pareto... if set, will return the pareto front instead of the top k
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

            if pareto:
                # does this loss + graph dominate anyone completely?
                # check which entries it dominates
                dominated_entries = []
                for i, (pareto_loss, pareto_graph) in enumerate(zip(top_losses, top_graphs)):
                    if pareto_loss >= loss and len(pareto_graph.node_dict) >= len(cgraph.node_dict):
                        dominated_entries.append(i)
                # keep only non dominated entries
                if len(dominated_entries) > 0:
                    top_consts = [c] + [top_consts[i] for i in range(len(top_consts)) if i not in dominated_entries]
                    top_losses = [loss] + [top_losses[i] for i in range(len(top_consts)) if i not in dominated_entries]
                    top_graphs = [cgraph] + [top_graphs[i] for i in range(len(top_consts)) if i not in dominated_entries]           

                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})


            elif loss <= loss_thresh:
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

            if pareto:
                # does this loss + graph dominate anyone completely?
                # check which entries it dominates
                dominated_entries = []
                for j, (pareto_loss, pareto_graph) in enumerate(zip(top_losses, top_graphs)):
                    if pareto_loss >= loss and len(pareto_graph.node_dict) >= len(cgraphs[i].node_dict):
                        dominated_entries.append(j)
                # keep only non dominated entries
                if len(dominated_entries) > 0:
                    top_consts = [c] + [top_consts[j] for j in range(len(top_consts)) if j not in dominated_entries]
                    top_losses = [loss] + [top_losses[j] for j in range(len(top_consts)) if j not in dominated_entries]
                    top_graphs = [cgraphs[i]] + [top_graphs[j] for j in range(len(top_consts)) if j not in dominated_entries]           

                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})



            elif loss <= loss_thresh:
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
                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses)})

    sort_idx = np.argsort(top_losses)
    top_graphs = [top_graphs[i] for i in sort_idx]
    top_consts = [top_consts[i] for i in sort_idx]
    top_losses = [top_losses[i] for i in sort_idx]
    
    ret = {
        'graphs' : top_graphs,
        'consts' : top_consts,
        'losses' : top_losses}

    return ret

def hierarchical_search(X:np.ndarray, n_outps: int, loss_fkt: callable, k: int, n_calc_nodes:int = 1, n_processes:int = 1, topk:int = 5, verbose:int = 0, opt_mode:str = 'grid', max_orders:int = 10000, max_time:float = 900.0, stop_thresh:float = -1.0, hierarchy_stop_thresh:float = -1.0, unique_loss:bool = True, pareto:bool = False, **params) -> dict:
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
        max_time... when this time (in seconds) has passed, no new orders are evaluated
        stop_thresh... if loss is lower than this, will stop evaluation (only for single process)
        hierarchy_stop_thresh... if loss of any top performer is lower than this, will stop after hierarchy step
        unique_loss... only take graph into topk if it has a totally new loss
        pareto... if set, will return the pareto front instead of the top k
    @Returns:
        dictionary with:
            graphs -> list of computational DAGs
            consts -> list of constants
            losses -> list of losses

    '''
    ret = {
        'graphs' : [],
        'consts' : [],
        'losses' : []}
    process_start_time = time.time()


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
        setup_time = time.time() - process_start_time
        if setup_time >= max_time:
            break

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
                if early_stop:
                    break

                consts, losses, ops = evaluate_build_order(order, m, n, k, X, loss_fkt, topk, opt_mode = opt_mode, start_time=process_start_time, max_time=max_time, pareto=pareto)
                
                if pareto:
                    top_losses += losses
                    top_consts += consts
                    top_ops += ops
                    top_orders += [order]*len(losses)
                    
                    complexities = [len(op) for op in top_ops]
                    pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                    if unique_loss:
                        pareto_losses = set()
                        take_idxs = []
                        for j in pareto_idxs:
                            v = np.round(top_losses[j], 15)
                            if v not in pareto_losses:
                                pareto_losses.add(v)
                                take_idxs.append(j)
                    else:
                        take_idxs = pareto_idxs
                    
                    top_losses = [top_losses[i] for i in take_idxs]
                    top_consts = [top_consts[i] for i in take_idxs]
                    top_ops = [top_ops[i] for i in take_idxs]
                    top_orders = [top_orders[i] for i in take_idxs]


                
                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})

                
                else:
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
                early_stop = np.any(np.array(top_losses) < stop_thresh)


        else:
            args = [[order, m, n, k, X, loss_fkt, topk, opt_mode, stop_thresh, process_start_time, max_time- setup_time, pareto] for order in orders]
            if verbose == 2:
                pbar = tqdm(args, total = len(args))
            else:
                pbar = args

            with ctx.Pool(processes=n_processes, initializer=init_process, initargs=(early_stop,)) as pool:
                pool_results = pool.starmap(evaluate_build_order, pbar)


            for i, (consts, losses, ops) in enumerate(pool_results):
                if early_stop:
                    break

                if pareto:
                    top_losses += losses
                    top_consts += consts
                    top_ops += ops
                    top_orders += [orders[i]]*len(losses)
                    
                    complexities = [len(op) for op in top_ops]
                    pareto_idxs = utils.get_pareto_idxs(top_losses, complexities)

                    if unique_loss:
                        pareto_losses = set()
                        take_idxs = []
                        for j in pareto_idxs:
                            v = np.round(top_losses[j], 15)
                            if v not in pareto_losses:
                                pareto_losses.add(v)
                                take_idxs.append(j)
                    else:
                        take_idxs = pareto_idxs
                    
                    top_losses = [top_losses[j] for j in take_idxs]
                    top_consts = [top_consts[j] for j in take_idxs]
                    top_ops = [top_ops[j] for j in take_idxs]
                    top_orders = [top_orders[j] for j in take_idxs]
                    
                    if verbose == 2:
                        pbar.set_postfix({'best_loss' : np.min(top_losses), 'n_pareto' : len(top_losses)})


                else:
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
                early_stop = np.any(np.array(top_losses) < stop_thresh)

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
    ret['graphs'] = top_graphs
    ret['consts'] = top_consts
    ret['losses'] = top_losses

    return ret



########################
# Sklearn Interface for Symbolic Regression Task
########################

class DAGRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for exhaustive search.
    '''

    def __init__(self, k:int = 1, n_calc_nodes:int = 5, max_orders:int = int(1e6), random_state:int = None, processes:int = 1, max_samples:int = 100, stop_thresh:float = 1e-20, mode : str = 'exhaustive', loss_fkt :DAG_Loss_fkt = R2_loss_fkt, max_time:float = 1800.0, positives:list = None, pareto:bool = False, **kwargs):
        '''
        @Params:
            k.... number of constants
            n_calc_nodes... number of possible intermediate nodes
            max_orders... maximum number of expression - skeletons in search
            random_state... for reproducibility
            processes... number of processes for multiprocessing
            max_samples... maximum number of samples on which to fit
            stop_thresh... threshold for early stopping, set to negative value if you dont want it
            mode... one of 'exhaustive' or 'hierarchical'
            loss_fkt... loss function class
            positives... marks which X are strictly positive
            pareto... flag for exhaustive search
        '''
        self.k = k
        self.n_calc_nodes = n_calc_nodes
        self.max_orders = max_orders
        self.max_samples = max_samples
        assert mode in ['exhaustive', 'hierarchical'], f'Search mode {mode} is not supported.'
        self.mode = mode
        self.stop_thresh = stop_thresh
        self.max_time = max_time
        self.processes = max(min(processes, multiprocessing.cpu_count()), 1)

        self.random_state = random_state
        self.loss_fkt = loss_fkt
        self.positives = positives
        self.pareto = pareto

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 1):
        '''
        Fits a model on given regression data.
        @Params:
            X... input data (shape n_samples x inp_dim)
            y... output data (shape n_samples)
            processes... number of processes for evaluation
        '''
        assert len(y.shape) == 1, f'y must be 1-dimensional (current shape: {y.shape})'

        self.positives = np.all(X > 0, axis = 0)

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
            'stop_thresh' : self.stop_thresh,
            'max_time' : self.max_time,
            'pareto' : self.pareto
        }
        if self.mode == 'hierarchical':
            res = hierarchical_search(**params)
        else:
            res = exhaustive_search(**params)
        self.results = res

        if len(res['graphs']) > 0:
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
        expr = self.cgraph.evaluate_symbolic(c = self.consts)[0]

        if self.positives is not None:
            transl_dict = {}
            for s in expr.free_symbols:
                idx = int(str(s).split('_')[-1])
                if self.positives[idx]:
                    transl_dict[s] = sympy.Symbol(f'x_{idx}', real = True, positive = True)
                else:
                    transl_dict[s] = sympy.Symbol(f'x_{idx}', real = True)
            expr = expr.subs(transl_dict)
        return expr

    def complexity(self):
        '''
        Complexity of expression (number of calculations)
        '''
        assert hasattr(self, 'cgraph'), 'No graph found yet. Call .fit first!'
        return self.cgraph.n_operations()
    
########################
# New: Polynomial Feature Regressor
# ########################
    
class BaseReg():
    '''
    Regressor based on Linear combination of polynomials and trigonometric functions
    '''
    def __init__(self, degree:int = 2, verbose:int = 0, random_state:int = 0, alpha:float = 0.0, max_terms:int = -1, interactions:int = 2, normalize:bool = False, **params):
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

class DAGRegressorPoly(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''
    Symbolic DAG-Search

    Sklearn interface for symbolic Regressor based on replacement strategies.
    '''

    def __init__(self, random_state:int = None, simpl_nodes:int = 3, topk:int = 1, max_orders:int = int(1e5), max_degree:int = 7, max_tree_size:int = 30, max_samples:int = 100, processes:int = 1, **kwargs):
        self.random_state = random_state
        self.processes = processes
        self.regr_search = self.regr_search = DAGRegressor(processes=self.processes, random_state = self.random_state)
        self.regr_poly = None
        self.simpl_nodes = simpl_nodes
        self.max_orders = max_orders
        self.topk = topk
        self.max_degree = max_degree
        self.max_tree_size = max_tree_size
        self.max_samples = max_samples

    def fit(self, X:np.ndarray, y:np.ndarray, verbose:int = 0):
        max_tree_size = self.max_tree_size
        fit_thresh = 1-(1e-8) # we consider everything above this as 'recovered'

        if self.random_state is not None:
            np.random.seed(self.random_state)
            

        # check for polynomial
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)
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

            for degree, score in zip(polydegrees, test_scores[:len(polydegrees)]):
                if score > 0.999:
                    break
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
                'stop_thresh' : 1e-20
            }
            res = exhaustive_search(**params)

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
                            expr = utils.round_floats(expr, round_digits = 3)
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
                            expr = utils.round_floats(expr, round_digits = 3)
                            expr = utils.simplify(expr)
                            exprs.append(expr)
                            ts = utils.tree_size(expr)
                            if verbose > 0:
                                print(f'DAG: {score} Size: {ts}')
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
                expr = utils.round_floats(self.regr_search.model(), round_digits = 5)
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

                

                if False and np.any(scores > 0.99):
                    # select smallest of the best models [not used]
                    idxs = np.where(scores > 0.99)[0]
                    exprs = [exprs[i] for i in idxs]
                    sizes = sizes[idxs]
                    self.expr = exprs[np.argmin(sizes)]
                    
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
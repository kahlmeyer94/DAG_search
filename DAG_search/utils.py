import stopit
import zss
import numpy as np
import scipy.stats
import sympy
import traceback
from DAG_search import config
import warnings
import networkx as nx

#####################################
# Pareto Front
#####################################

def get_pareto_idxs(obj1, obj2):
    M = np.column_stack([obj1, obj2])
    ret = []
    for i, p in enumerate(M):
        is_dominated = (np.all(M <= p, axis = 1) & np.any(M != p, axis = 1)).sum() > 1
        if not is_dominated:
            ret.append(i)
    return np.array(ret)

#####################################
# Gradient estimation
#####################################

def est_gradient(reg, X, fx = None, eps = 1e-5):
    
    X_tmp = []
    for i in range(X.shape[1]):
        x_tmp = X.copy()
        x_tmp[:, i] += eps
        X_tmp.append(x_tmp)
        x_tmp = X.copy()
        x_tmp[:, i] -= eps
        X_tmp.append(x_tmp)
    X_tmp = np.concatenate(X_tmp, axis=0)

    grad_X = reg.predict(X_tmp)
    if fx is None:
        val_X = reg.predict(X)
    else:
        val_X = fx

    N = len(X)
    f_ = []
    for i in range(X_tmp.shape[1]):
        f_h_pos = grad_X[i*2*N : i*2*N + N]
        f_h_neg = grad_X[i*2*N + N : (i+1)*2*N]

        v1 = (f_h_pos - val_X)/eps
        v2 = (val_X - f_h_neg)/eps
        v = 0.5*(v1 + v2)
        f_.append(v)
    f_ = np.column_stack(f_)
    return f_

#####################################
# Locality
#####################################

def get_components(A):
    # create graph
    graph = nx.Graph()

    # add nodes
    for i in range(len(A)):
        graph.add_node(i)


    # add edges
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i, j] > 0:
                graph.add_edge(i, j)
    return nx.number_connected_components(graph)

#####################################
# Mixability
#####################################

def get_subexprs_sympy(expr) -> list:
    '''
    Splits an expression into subexpressions.
    @Params:
        expr... sympy expression
    @Returns:
        list of all subexpressions according to sympy
    '''

    func = expr.func
    children = expr.args
    subs = [expr]
    for c in children:
        subs += get_subexprs_sympy(c)
    return subs
    
def insert_subexprs(expr1, expr2) -> list:
    '''
    Creates all subexpressions that are created
    when placing expr2 and every node of expr1.

    @Params:
        expr1... sympy expression
        expr2... sympy expression

    @Returns:
        List of sympy expression
    '''

    # insert expr2 at every position of expr1
    func = expr1.func
    children = expr1.args
    if isinstance(expr2, sympy.Number):
        ret = [expr2]
    else:
        ret = [expr2]
    for i in range(len(children)):
        child_ret = insert_subexprs(children[i], expr2)
        for child_expr in child_ret:
            expr1_c = expr1
            args_list = [expr1_c.args[idx] for idx in range(len(expr1_c.args))]
            args_list[i] = child_expr
            expr1_c = expr1_c.func(*tuple(args_list))
            ret.append(expr1_c)
    return ret

def mix_error(expr, population : list, X : np.ndarray, y : np.ndarray, max_error : int = 1000) -> float:
    '''
    Mixability as defined by us.
    Expected performance when using an expression as new subtree.

    @Params:
        expr... sympy expression
        population... list of sympy expressions
        X... input data (shape n_samples x dim)
        y... output data (length n_samples)
        max_error... errors are capped at this value

    @Returns
        Average error when using expr as new subtree
    '''


    x_symbs = [sympy.Symbol(f'x_{i}', real = True) for i in range(X.shape[1])]
    errors = []
    if len(y.shape) == 1:
        ret_shape = (len(y), 1)
    else:
        ret_shape = y.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for expr_pop in population:
            insert_exprs = insert_subexprs(expr_pop, expr) # insert expr at expr_pop
            for expr_insert in insert_exprs:
                valid = True
                
                try:
                    with stopit.ThreadingTimeout(2.0, swallow_exc=False) as to_ctx_mgr:
                        assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
                        func = sympy.lambdify(*x_symbs, expr_insert)
                        pred = func(X)*np.ones(ret_shape)
                    if to_ctx_mgr:
                        valid = True
                    else:
                        valid = False
                except (TypeError, KeyError, AttributeError, RecursionError, stopit.utils.TimeoutException):
                    valid = False
                
                if valid and np.all(np.isreal(pred)) and not np.any(np.isnan(pred)):
                    mse = min(np.mean(np.abs(pred - y)), max_error)
                    errors.append(mse)
                else:
                    errors.append(max_error)
    return np.mean(errors)

def mean_confidence_interval(data : list, confidence : float = 0.95) -> tuple:
    '''
    Confidence interval for the mean of data.
    [found here: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data]

    @Params:
        data... list of numbers where we want to estimate the mean
        confidence... size of confidence interval

    @Returns:
        mean, lower, upper for confidence interval
    '''
    # 
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


#####################################
# Symbolic Checks
#####################################

def simplify(expr, timeout:float = 5.0):
    '''
    @Params:
        expr... sympy expression
        timeout... number of seconds after which to abort

    @Returns:
        simplified sympy expression
    '''

    try:
        expr_simp = None
        with stopit.ThreadingTimeout(timeout, swallow_exc=False) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            expr_simp = sympy.simplify(expr)

        if to_ctx_mgr:
            return expr_simp
        else:
            return expr
    except (TypeError, KeyError, AttributeError, RecursionError, stopit.utils.TimeoutException):
        return expr

def tree_size(expr) -> int:
    '''
    Counts number of nodes in expression tree.

    @Params:
        expr... sympy expression

    @Returns:
        number of nodes in expression tree
    '''

    children = expr.args
    return 1 + sum([tree_size(c) for c in children])

def is_const(expr, timeout:float = 5.0) -> bool:
    '''
    Checks whether sympy expression is a constant.

    @Params:
        expr... sympy expression
        timeout... number of seconds after which to abort

    @Returns:
        True, if expression could successfully be reduced to a constant
    '''

    ret = False
    try:
        with stopit.ThreadingTimeout(timeout, swallow_exc=False) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING
            ret = expr.is_constant()
        if to_ctx_mgr:
            return ret
        else:
            return (len(expr.free_symbols) == 0)
    except (TypeError, KeyError, AttributeError, RecursionError, stopit.utils.TimeoutException):
        return (len(expr.free_symbols) == 0)  
    
def symb_eq(expr_est, expr_true) -> bool:
    '''
    Checks symbolic equivalence of expressions.

    @Params:
        expr_est... sympy expression
        expr_true... sympy expression

    @Returns:
        True if both expressions are equivalent.
    '''

    try:
        

        # 0. make sure all constants have same properties
        transl_dict = {}
        for s_est in expr_est.free_symbols:
            for s_true in expr_true.free_symbols:
                if str(s_est) == str(s_true):
                    transl_dict[s_true] = s_est
        expr_true_tmp = expr_true.subs(transl_dict)
        expr_est_tmp = round_floats(simplify(expr_est))

        # If expression is absurdly large: do not try
        if tree_size(expr_est_tmp) > 50:
            return False

        # 1. difference reduces to constant
        expr_diff = round_floats(simplify(expr_true_tmp - expr_est_tmp))

        if expr_diff is None:
            return False
        #expr_diff = round_floats(expr_diff)
        if is_const(expr_diff) and not (sympy.nan == expr_diff or sympy.oo == expr_diff):
            return True
        else:
            # 2. ratio reduces to constant which is not 0
            expr_ratio = round_floats(simplify(expr_true_tmp/expr_est_tmp))

            if expr_ratio is None:
                return False
            if (expr_ratio != 0) and is_const(expr_ratio) and not (sympy.nan == expr_ratio or sympy.oo == expr_ratio):
                return True
            else:
                return False
    except Exception as e:
        traceback.print_exc()
        # all kind of exotic sympy exceptions can occur here.
        return False

def round_floats(ex1, round_digits:int = 3, max_v:int = np.inf):
    '''
    Rounds floats within sympy expression.

    @Params:
        ex1... sympy expression
        max_v... numbers greater are set to infinity
    
    @Returns:
        sympy expression
    '''

    ex2 = ex1.evalf()
    found = True
    max_rounds = 3
    n_rounds = 0

    while found and n_rounds < max_rounds:
        n_rounds += 1
        found = False
        try:
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Number):
                    if abs(a) > max_v:
                        if a > 0:
                            ex2 = ex2.subs(a, sympy.oo)
                        else:
                            ex2 = ex2.subs(a, -sympy.oo)
                        ex2 = sympy.cancel(ex2)
                    elif abs(round(a) - float(a)) <= 1/(10**round_digits):
                        found = True
                        ex2 = ex2.subs(a, sympy.Integer(round(a)))
                    else:
                        ex2 = ex2.subs(a, sympy.Float(round(a, round_digits), round_digits))
        except (TypeError, ValueError):
            found = False
        ex1 = ex2
    return ex1

def jaccard_idx(expr1, expr2):
    tmp_expr1 = simplify(round_floats(expr1.evalf()))
    tmp_expr2 = simplify(round_floats(expr2.evalf()))
    S0 = set([str(subexpr) for subexpr in get_subexprs_sympy(tmp_expr1)])
    S1 = set([str(subexpr) for subexpr in get_subexprs_sympy(tmp_expr2)])
    return len(S0&S1)/len(S0|S1)


#####################################
# Symbolic Distance Measures
#####################################

# Symbolic distance DFS comparison
def get_dfs(graph, idx):
    children, op = graph.node_dict[idx]
    ret = []
    if len(children) == 0:
        if op.startswith('c'):
            ret = ['const']
        else:
            ret = [op]
    else:
        ret = [op]
        for child in children:
            ret += get_dfs(graph, child)
    return ret

def lcsubstring_length(a, b):
    # found here: https://stackoverflow.com/questions/24547641/python-length-of-longest-common-subsequence-of-lists
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    longest = 0
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            if ca == cb:
                length = table[i][j] = table[i - 1][j - 1] + 1
                longest = max(longest, length)
    return longest

def dfs_ratio(graph1, graph2):
    
    outp_idx1 = graph1.inp_dim + graph1.n_consts
    outp_idx2 = graph2.inp_dim + graph2.n_consts

    dfs1 = get_dfs(graph1, outp_idx1)
    dfs2 = get_dfs(graph2, outp_idx2)
    return (lcsubstring_length(dfs1, dfs2) - 1)/max(len(dfs1), len(dfs2))


# Symbolic distance subexpression comparison
def get_subexprs(graph):
    X = [sympy.symbols(f'x_{i}', real = True) for i in range(graph.inp_dim)]
    c = [sympy.symbols(f'c_{i}', real = True) for i in range(graph.n_consts)]
    ret_nodes = graph.outp_nodes

    # we assume that eval order is valid
    k = graph.inp_dim + graph.n_consts

    res_dict = {i : None for i in graph.node_dict}
    # inputs        
    for i in range(graph.inp_dim):
        res_dict[i] = X[i]

    for i, const in enumerate(c):
        res_dict[i + graph.inp_dim] = const

    # others
    for i in graph.eval_order[k:]:
        children, node_op = graph.node_dict[i]
        node_op = config.NODE_OPS_SYMB[node_op]
        child_results = [res_dict[j] for j in children]

        node_result = child_results[0]

        if len(child_results) > 1:
            for j in range(1, len(child_results)):
                node_result = node_op(node_result, child_results[j])
        else:
            node_result = node_op(node_result)

        res_dict[i] = node_result
    
    ret_dict = {}
    for i in res_dict:
        ret_dict[i] = str(res_dict[i])
    return ret_dict

def get_depths(graph):
    k = graph.inp_dim + graph.n_consts
    depth_dict = {i : 0 for i in graph.node_dict}

    for i in graph.eval_order[k:]:
        children, node_op = graph.node_dict[i]
        node_depth = max([depth_dict[j] for j in children])
        if node_op != '=':
            node_depth += 1
        depth_dict[i] = node_depth
    return depth_dict

def subexpr_ratio(graph1, graph2):
    subexprs1 = get_subexprs(graph1)
    subexprs2 = get_subexprs(graph2)
    
    
    depths1 = get_depths(graph1)
    depths2 = get_depths(graph2)
    d1 = graph1.depth()
    d2 = graph2.depth()
    
    
    exprs1 = set([subexprs1[i] for i in subexprs1])
    exprs2 = set([subexprs2[i] for i in subexprs2])
    common_exprs = list(exprs1 & exprs2)
    
    scores = []
    for expr in common_exprs:
        
        # find depth in graph1
        tmp = []
        for i1 in subexprs1:
            if subexprs1[i1] == expr:
                tmp.append(depths1[i1])
        depth1 = min(tmp)
        
        
        # find depth in graph2
        tmp = []
        for i1 in subexprs2:
            if subexprs2[i1] == expr:
                tmp.append(depths2[i1])
        depth2 = min(tmp)

        score = min(depth1/(d1 + 1), depth2/(d2 + 1))
        scores.append(score)
        
    if len(scores) == 0:
        return 0.0
    else:
        return max(scores)
    

# Zhang-Shasha tree edit distance
# https://epubs.siam.org/doi/10.1137/0218082

def graph2trees(graph):
    trees = []
    for i in range(graph.outp_dim):
        root_idx = i + graph.inp_dim + graph.n_consts
        trees.append(get_tree(graph, root_idx))
    return trees

def get_tree(graph, idx):
    children, op = graph.node_dict[idx]
    if len(children) == 0:
        return zss.Node(op)
    else:
        ret = zss.Node(op)
        for child in children:
            ret.addkid(get_tree(graph, child))
        return ret
    
def graph_edit_distance(graph1, graph2):
    tree1 = graph2trees(graph1)[0]
    tree2 = graph2trees(graph2)[0]
    return zss.simple_distance(tree1, tree2)


def expr2tree(expr):
    op = str(expr.func)
    children = expr.args
    if len(children) == 0:
        return zss.Node(op)
    else:
        ret = zss.Node(op)
        for child in children:
            ret.addkid(expr2tree(child))
        return ret

def expr_edit_distance(expr1, expr2):
    tree1 = expr2tree(expr1)
    tree2 = expr2tree(expr2)
    return zss.simple_distance(tree1, tree2)
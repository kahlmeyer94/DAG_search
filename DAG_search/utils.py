import stopit
import zss
import sympy
import traceback
from DAG_search import config

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
        expr_est_tmp = round_floats(expr_est)

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

def round_floats(ex1, max_v:int = 1000):
    '''
    Rounds floats within sympy expression.

    @Params:
        ex1... sympy expression
        max_v... numbers greater are set to infinity
    
    @Returns:
        sympy expression
    '''

    ex2 = ex1
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
                    elif abs(round(a) - float(a)) <= 0.01:
                        found = True
                        ex2 = ex2.subs(a, sympy.Integer(round(a)))
                    else:
                        ex2 = ex2.subs(a, sympy.Float(round(a, 3),3))
        except (TypeError, ValueError):
            found = False
        ex1 = ex2
    return ex1


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
    
def edit_distance(graph1, graph2):
    tree1 = graph2trees(graph1)[0]
    tree2 = graph2trees(graph2)[0]
    return zss.simple_distance(tree1, tree2)
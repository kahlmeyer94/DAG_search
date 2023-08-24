import stopit
import sympy
import traceback

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

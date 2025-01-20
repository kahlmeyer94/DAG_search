from itertools import combinations
import numpy as np
import sympy

class DA_Buckingham():
    ######################################################
    # Dimensional Analysis using Buckinghams Pi - Theorem
    ######################################################
    def __init__(self): 
        pass

    def fit(self, D, X):
        '''
        @Params:
            D.. units matrix, shape (n_vars) x n_units,
            X... data matrix, shape n_data x n_vars

        @Returns:
            X_red, translation dictionary   
        '''
        M = np.array(D).T

        U = sympy.Matrix(M).nullspace()
        U = np.column_stack([np.array(u).astype(int) for u in U])
            
        
        # create new data
        X_new = []
        transl_dict = {}
        for i in range(U.shape[1]):
            xi = np.ones(len(X))
            transl_dict[i] = ''
            for j in range(U.shape[0]):
                if U[j, i] != 0:
                    xi = xi*(X[:, j]**(U[j, i]))
                    transl_dict[i] += f'(x_{j}**({U[j, i]}))*'
            if len(transl_dict[i]) > 0:
                transl_dict[i] = transl_dict[i][:-1]
            X_new.append(xi)
        X_new = np.column_stack(X_new)
            
        return X_new, transl_dict

    def translate(self, expr_red, transl_dict):
        '''
        Translates the dimensionless expression back to dimensions.

        @Params:
            expr_red... dimensionless expression
            transl_dict... dictionary from dimensional analysis
        '''
        expr_str = str(expr_red).replace('x_', 'v_')
        for i in range(len(transl_dict)):
            expr_str = expr_str.replace(f'v_{i}', f'({transl_dict[i]})')
        return sympy.sympify(expr_str)
            
class DA_Feynman():
    ######################################################
    # Dimensional Analysis using Buckinghams Pi - Theorem
    # as performed by AIFeynman
    ######################################################
    def __init__(self): 
        pass

    def get_p(self, M, b):
        # copied from here: https://github.com/SJ001/AI-Feynman/blob/master/aifeynman/getPowers.py#L27
        # tries to find a sparse solution vector p
        rand_drop_cols = np.arange(0, M.shape[1], 1)
        rand_drop_rows = np.arange(0, M.shape[0], 1)
        rand_drop_rows = np.flip(rand_drop_rows)
        rank = np.linalg.matrix_rank(M)
        d_cols = list(combinations(rand_drop_cols,M.shape[1]-rank))
        d_rows = list(combinations(rand_drop_rows, M.shape[0]-rank))
        
        # find out which rows and columns to remove
        for i in d_cols:
            M1 = M.copy()
            M1 = np.delete(M1, i, 1)
            M1 = np.transpose(M1)
            for j in d_rows:
                M2 = M1.copy()
                M2 = np.delete(M2, j, 1)
                if np.linalg.det(M2)!=0:
                    solved_M = np.transpose(M2)
                    indices_sol = j
                    indices_powers = i
                    break
        
        # remove and find solution in reduced problem
        solved_b = np.delete(b, indices_sol)
        params = np.linalg.solve(solved_M, solved_b)
        
        # reconstruct solution for original problem
        sol = []
        for i in range(M.shape[1]):
            if i in indices_powers:
                sol = sol + [0]
            else:
                sol = sol + [params[0]]
                params = np.delete(params,0)
        
        # this is the solution
        return np.array(sol)

    def get_U(self, M):
        return sympy.Matrix(M).nullspace()

    def fit(self, D, X, y):
        '''
        @Params:
            D.. units matrix, shape (n_vars+1) x n_units, last row is for dependent variable
            X... data matrix, shape n_data x n_vars
            y... dependent data array, shape n_data

        @Returns:
            X_red, y_red, translation dictionary   
        '''
        b = np.array(D[-1]).reshape(-1, 1)
        M = np.array(D[:-1]).T

        try:
            # get vector for dependent variable
            p = self.get_p(M, b)
            assert np.allclose(M@p.reshape(-1, 1) - b, 0)
        
            # get vectors for independent variables
            U = self.get_U(M)
            assert len(U) > 0
            U = np.column_stack([np.array(u).astype(int) for u in U])
            assert np.allclose(M@U, 0)
        
        
            # create new data
            X_new = []
            transl_dict = {}
            for i in range(U.shape[1]):
                xi = np.ones(len(X))
                transl_dict[i] = ''
                for j in range(U.shape[0]):
                    if U[j, i] != 0:
                        xi = xi*(X[:, j]**(U[j, i]))
                        transl_dict[i] += f'(x_{j}**({U[j, i]}))*'
                if len(transl_dict[i]) > 0:
                    transl_dict[i] = transl_dict[i][:-1]
                X_new.append(xi)
            X_new = np.column_stack(X_new)
            
            y_new = y.copy()
            transl_dict['y'] = 'y*'
            for i in range(len(p)):
                if p[i] != 0:
                    y_new *= X[:, i]**(-p[i])
                    transl_dict['y'] += f'(x_{i})**({-p[i]})*'
            transl_dict['y'] = transl_dict['y'][:-1] 
        except AssertionError:
            return None

        return X_new, y_new, transl_dict

    def translate(self, expr_red, transl_dict):
        '''
        Translates the dimensionless expression back to dimensions.

        @Params:
            expr_red... dimensionless expression
            transl_dict... dictionary from dimensional analysis
        '''
        try:
            expr_str = str(expr_red).replace('x_', 'v_')
            for i in range(len(transl_dict)-1):
                expr_str = expr_str.replace(f'v_{i}', f'({transl_dict[i]})')
            sols = sympy.solve(sympy.sympify(f"({transl_dict['y']}) - ({expr_str})"), sympy.Symbol('y'))
            assert len(sols) == 1
            return sols[0]
        except AssertionError:
            return None

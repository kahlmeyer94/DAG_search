'''
The computational graph class
'''

import numpy as np
import config
import copy
import warnings
import sympy
from sympy import simplify
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch


class CompGraph():

    def __init__(self, m, n, k, node_dict = {}, expr = None):
        '''
        m... input dimension
        n... output dimension
        k... number of constants
        '''
        
        self.inp_dim = m
        self.outp_dim = n
        self.n_consts = k

        self.node_dict = node_dict # key = node_idx, value = list of children , operation type

        self.R = None # reachability matrix (number of paths from node i to node j)
        self.A = None # adjacency matrix
        self.eval_order = None # index order in which nodes are evaluated

        self.outp_nodes = []
        self.inp_nodes = []
        self.successors = {}
        self.predecessors = {}

        if len(self.node_dict) > 0:
            self.update_stats()

    def n_nodes(self):
        return len(self.node_dict)

    def n_edges(self):
        ret = 0
        for i in self.successors:
            ret += len(self.successors[i])
        return ret

    def n_operations(self):
        return len(self.node_dict) - (self.inp_dim + self.n_consts)

    def depth(self):
        k = self.inp_dim + self.n_consts
        depth_dict = {i : 0 for i in self.node_dict}

        for i in self.eval_order[k:]:
            children, node_op = self.node_dict[i]
            node_depth = max([depth_dict[j] for j in children])
            if node_op != '=':
                node_depth += 1
                
            depth_dict[i] = node_depth

        return max([depth_dict[i] for i in self.outp_nodes])

    def n_subexps(self):
        m, n, k = self.inp_dim, self.outp_dim, self.n_consts
        counts = {i : 0 for i in self.node_dict if i >= m + n + k}
        for i in self.node_dict:
            parents, _ = self.node_dict[i]
            for p in parents:
                if p in counts:
                    counts[p] += 1
        return (np.array([counts[i] for i in counts]) > 1).sum()

    def copy(self):
        ret = CompGraph(self.inp_dim, self.outp_dim, self.n_consts)
        ret.node_dict = copy.deepcopy(self.node_dict)

        ret.update_stats()
        return ret

    def update_stats(self):

        
        # update A and R
        n_nodes = len(self.node_dict)
        self.A = np.zeros((n_nodes, n_nodes))
        self.R = np.eye(n_nodes)
        for i in range(n_nodes):
            children = self.node_dict[i][0]
            for j in children:
                self.A[j, i] += 1
                self.R += np.outer(self.R[:, i], self.R[j,:])

        # update eval_order
        self.eval_order = []
        for order in self.get_eval_order():
            self.eval_order = self.eval_order + order


        self.outp_nodes = [self.inp_dim + self.n_consts + i for i in range(self.outp_dim)]
        self.inp_nodes = list(range(self.inp_dim + self.n_consts))

        self.successors = {i : [] for i in self.node_dict}
        self.predecessors = {i : [] for i in self.node_dict}
        for i in self.node_dict:
            for j in self.node_dict[i][0]:
                self.successors[j].append(i)
                self.predecessors[i].append(j)

    def get_eval_order(self):
        # get layers of nodes

        eval_order = []
        eval_idx = set() # already evaluated indices
        remain_idx = set(range(len(self.node_dict))) # yet to be evaluated

        # inputs
        round_list = []
        for i in range(self.inp_dim + self.n_consts):
            round_list.append(i)
            eval_idx.add(i)
            remain_idx.remove(i)
        eval_order.append(round_list)

        while len(remain_idx) > 0:
            next_idx = []
            for i in remain_idx:
                # are all children evaluated?
                valid = True
                for j in self.node_dict[i][0]:
                    if j not in eval_idx:
                        valid = False
                        break
                if valid:
                    next_idx.append(i)

            for i in next_idx:
                eval_idx.add(i)
                remain_idx.remove(i)
            eval_order.append(next_idx)
        return eval_order

    def get_subexps(self):
        eval_order = self.get_eval_order()
        n_paths = (self.R[self.outp_nodes] > 0).sum(axis=0)
        node_ids = np.arange(0, len(self.node_dict), 1)
        subexp_ids = []
        for i, nodes in enumerate(eval_order):
            mask = (n_paths[nodes] > 1)
            if np.any(mask):
                subexp_ids.append(np.array(nodes)[mask])
            else:
                break
        subexps = [self.evaluate_symbolic(ret_nodes = order) for order in subexp_ids]
        return subexps

    def max_subexp_depth(self, eval_order = None):
        if eval_order is None:
            eval_order = self.get_eval_order()
        max_depth = len(eval_order)
        n_paths = (self.R[self.outp_nodes] > 0).sum(axis=0)
        current_depth = 0
        for i, nodes in enumerate(eval_order):
            mask = (n_paths[nodes] > 1)
            if np.any(mask):
                current_depth = (i+1)
        return current_depth

    def subexp_ratio(self):
        eval_order = self.get_eval_order()
        max_depth = len(eval_order)
        return self.max_subexp_depth(eval_order)/max_depth

    # Evaluation
    
    def get_gradient(self, X, c):
        N = X.shape[0]
        k = self.inp_dim + self.n_consts

        grad_dict = {i : None for i in self.node_dict}
        X_tensor = torch.tensor(X, requires_grad = True).double()

        c_tensor = (torch.ones((N, len(c))) * torch.tensor(c)).double()
        c_tensor.requires_grad = True

        # inputs        
        for i in range(self.inp_dim):
            grad_dict[i] = X_tensor[:, i]

        for i in range(self.n_consts):
            grad_dict[i + self.inp_dim] = c_tensor[:, i]

        # others
        for i in self.eval_order[k:]:
            children, op = self.node_dict[i]

            node_op_pt = config.NODE_OPS_PYTORCH[op]
            child_results_pt = torch.stack([grad_dict[j] for j in children])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if child_results_pt.shape[0] > 1:
                    grad_result = node_op_pt(child_results_pt, axis=0)
                else:
                    grad_result = node_op_pt(child_results_pt[0])
                grad_dict[i] = grad_result


        h_X = torch.column_stack([grad_dict[i] for i in self.outp_nodes])

        if h_X.requires_grad:
            # shape: outps x N x inps
            grad_X = []
            for i in range(h_X.shape[1]):
                part_grad = torch.autograd.grad(h_X[:,i], X_tensor, grad_outputs=h_X[:,i].data.new(h_X[:,i].shape).fill_(1), create_graph=True, allow_unused=True)[0]
                if part_grad is None:
                    part_grad = np.zeros((h_X.shape[0], X.shape[1]))
                else:
                    part_grad = part_grad.detach().numpy()
                grad_X.append(part_grad)    
            grad_X = np.stack(grad_X)

            grad_c = []
            for i in range(h_X.shape[1]):
                part_grad = torch.autograd.grad(h_X[:,i], c_tensor, grad_outputs=h_X[:,i].data.new(h_X[:,i].shape).fill_(1), create_graph=True, allow_unused=True)[0]
                if part_grad is None:
                    part_grad = np.zeros((h_X.shape[0], len(c)))
                else:
                    part_grad = part_grad.detach().numpy()
                grad_c.append(part_grad)
            grad_c = np.stack(grad_c)

        else:
            grad_X = np.zeros((h_X.shape[1], h_X.shape[0], X.shape[1]))
            grad_c = np.zeros((h_X.shape[1], h_X.shape[0], len(c)))
        return grad_X, grad_c

    def evaluate_old(self, X, c, return_grad = False):
        '''
        X... N x m matrix
        c... array of length n_consts
        return_grad... if true, will return derivative of output wrt. input
        '''
        assert len(self.node_dict) >= self.inp_dim + self.n_consts + self.outp_dim, 'Node dict not initialized'
        assert len(c.shape) <= 2, 'Constants must be either 1D (single) or 2D (multiple)'

        
        # we assume that eval order is valid

        N = X.shape[0]
        k = self.inp_dim + self.n_consts

        res_dict = {i : None for i in self.node_dict}
        grad_dict = {i : None for i in self.node_dict}


        #X_stacked = np.stack([X]*r)
        X_tensor = torch.tensor(X, requires_grad = True).double()

        # inputs        
        for i in range(self.inp_dim):
            res_dict[i] = X[:,i]
            grad_dict[i] = X_tensor[:,i]

        for i in range(self.n_consts):
            res_dict[i + self.inp_dim] = c[i]*np.ones(N)
            grad_dict[i + self.inp_dim] = c[i]*torch.ones(N)

        # others
        for i in self.eval_order[k:]:
            children, op = self.node_dict[i]

            node_op = config.NODE_OPS[op]
            child_results = np.stack([res_dict[j] for j in children])

            node_op_pt = config.NODE_OPS_PYTORCH[op]
            child_results_pt = torch.stack([grad_dict[j] for j in children])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if child_results.shape[0] > 1:
                    node_result = node_op(child_results, axis=0)
                    grad_result = node_op_pt(child_results_pt, axis=0)
                else:
                    node_result = node_op(child_results[0])
                    grad_result = node_op_pt(child_results_pt[0])

                res_dict[i] = node_result
                grad_dict[i] = grad_result

        final_result = np.column_stack([res_dict[i] for i in self.outp_nodes])

    
        if return_grad:

            h_X = torch.column_stack([grad_dict[i] for i in self.outp_nodes])

            if h_X.requires_grad:
                # shape: outps x N x inps
                h_X_grad = []
                for i in range(h_X.shape[1]):
                    part_grad = torch.autograd.grad(h_X[:,i], X_tensor, grad_outputs=h_X[:,i].data.new(h_X[:,i].shape).fill_(1), create_graph=True)[0]
                    h_X_grad.append(part_grad.detach().numpy())
                h_X_grad = np.stack(h_X_grad)
                #h_X_grad = torch.autograd.grad(h_X, X_tensor, grad_outputs=h_X.data.new(h_X.shape).fill_(1), create_graph=True)[0].detach().numpy()
            else:
                h_X_grad = np.zeros((h_X.shape[1], h_X.shape[0], X.shape[1]))
                #h_X_grad = np.zeros(X.shape)
            return final_result, h_X_grad
        else:
            return final_result

    def evaluate(self, X, c, return_grad = False):
        '''
        X... N x m matrix
        c... array of length n_consts
        return_grad... if true, will return derivative of output wrt. input
        '''
        assert len(self.node_dict) >= self.inp_dim + self.n_consts + self.outp_dim, 'Node dict not initialized'
        assert len(c.shape) <= 2, 'Constants must be either 1D (single) or 2D (multiple)'

        if len(c.shape) == 2:
            r = c.shape[0]
            vec = True
        else:
            c = c.reshape(1, -1)
            r = 1
            vec = False

        # we assume that eval order is valid

        N = X.shape[0]
        k = self.inp_dim + self.n_consts

        res_dict = {i : None for i in self.node_dict}
        grad_dict = {i : None for i in self.node_dict}


        X_stacked = np.stack([X]*r)
        X_tensor = torch.tensor(X_stacked, requires_grad = True).double()

        # inputs        
        for i in range(self.inp_dim):
            res_dict[i] = X_stacked[:, :, i]
            grad_dict[i] = X_tensor[:, :, i]

        for i in range(self.n_consts):
            res_dict[i + self.inp_dim] = np.column_stack([c[:, i]]*N)
            grad_dict[i + self.inp_dim] = torch.as_tensor(res_dict[i + self.inp_dim])

        # others
        for i in self.eval_order[k:]:
            children, op = self.node_dict[i]

            node_op = config.NODE_OPS[op]
            child_results = np.stack([res_dict[j] for j in children])

            node_op_pt = config.NODE_OPS_PYTORCH[op]
            child_results_pt = torch.stack([grad_dict[j] for j in children])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if child_results.shape[0] == 2:
                    node_result = node_op(child_results[0], child_results[1])
                    grad_result = node_op_pt(child_results_pt[0], child_results_pt[1])
                else:
                    assert child_results.shape[0] == 1
                    node_result = node_op(child_results[0])
                    grad_result = node_op_pt(child_results_pt[0])

                res_dict[i] = node_result
                grad_dict[i] = grad_result


        final_result = np.stack([res_dict[i] for i in self.outp_nodes])
        final_result = np.transpose(final_result, [1, 2, 0])

        if return_grad:

            h_X = torch.stack([grad_dict[i] for i in self.outp_nodes])
            h_X = torch.permute(h_X, (1, 2, 0))

            if h_X.requires_grad:
                # shape: r x n x N x inps
                h_X_grad = []
                for i in range(h_X.shape[2]):
                    part_grad = torch.autograd.grad(h_X[:, :, i], X_tensor, grad_outputs=h_X[:,:,i].data.new(h_X[:,:,i].shape).fill_(1), create_graph=True)[0]
                    h_X_grad.append(part_grad.detach().numpy())
                h_X_grad = np.stack(h_X_grad)
                h_X_grad = np.transpose(h_X_grad, (1, 0, 2, 3))
            else:
                h_X_grad = np.zeros((h_X.shape[0], h_X.shape[2], h_X.shape[1], X.shape[1]))

            if not vec:
                return final_result[0], h_X_grad[0]
            else:
                return final_result, h_X_grad
        else:

            if not vec:
                return final_result[0]
            else:
                return final_result

    def evaluate_symbolic(self, X = None, c = None, ret_nodes = None):
        '''
        X... list of sympy symbols
        c... iterable of length n_consts
        '''
        if X is None:
            X = [sympy.symbols(f'x_{i}', real = True) for i in range(self.inp_dim)]
        if c is None or (len(c) == 0):
            c = [sympy.symbols(f'c_{i}', real = True) for i in range(self.n_consts)]
        if ret_nodes is None:
            ret_nodes = self.outp_nodes
        
        if len(self.node_dict) < self.inp_dim + self.n_consts + self.outp_dim:
            return None

        # we assume that eval order is valid
        k = self.inp_dim + self.n_consts

        res_dict = {i : None for i in self.node_dict}
        # inputs        
        for i in range(self.inp_dim):
            res_dict[i] = X[i]

        for i, const in enumerate(c):
            res_dict[i + self.inp_dim] = const

        # others
        for i in self.eval_order[k:]:
            children, node_op = self.node_dict[i]
            node_op = config.NODE_OPS_SYMB[node_op]
            child_results = [res_dict[j] for j in children]

            node_result = child_results[0]

            if len(child_results) > 1:
                for j in range(1, len(child_results)):
                    node_result = node_op(node_result, child_results[j])
            else:
                node_result = node_op(node_result)

            res_dict[i] = node_result
        return [res_dict[i] for i in ret_nodes]


def is_equal(cgraph1:CompGraph, cgraph2:CompGraph, consts1:np.ndarray = None, consts2:np.ndarray = None, use_sympy:bool = False):
    '''
    Checks if two graphs are equal

    @Params:
        cgraph1... computational graph
        cgraph2... computational graph
        consts1... (optional) constant values for graph1
        consts2... (optional) constant values for graph2
        use_sympy... if true, compares symbolic expressions

    @Returns:
        bool
    '''

    if not use_sympy:

        dict1 = cgraph1.node_dict
        dict2 = cgraph2.node_dict
        
        
        if len(dict1) != len(dict2):
            return False
        
        k1 = sorted(dict1.keys())
        k2 = sorted(dict2.keys())
        
        for i1, i2 in zip(k1, k2):
            if i1 != i2:
                return False
            else:
                l1, op1 = dict1[i1]
                l2, op2 = dict2[i2]
                if (sorted(l1), op1) != (sorted(l2), op2):
                    return False
                
        return True
    else:
        tmp1 = cgraph1.evaluate_symbolic(c = consts1)
        tmp2 = cgraph2.evaluate_symbolic(c = consts2)
        return np.all([(simplify(exp1-exp2) == 0) for exp1, exp2 in zip(tmp1, tmp2)])

def get_sympy_dag(expr, tree_dict = None):
    if tree_dict is None:
        tree_dict = {}
    idx = -1
    for i in tree_dict:
        if tree_dict[i][0] == expr:
            idx = i
            break
    if idx < 0:
        # new node
        idx = len(tree_dict)
    
        func = expr.func
        children = expr.args
        
        if func is sympy.core.power.Pow and int(children[1]) != float(children[1]):
            func = sympy.exp
            children = [float(children[1])*sympy.log(children[0])]
        
        if func is sympy.core.mul.Mul and children[0] == -1:
            func = 'neg'
            child_exp = children[1]
            for i in range(2, len(children)):
                child_exp = child_exp * children[i]
            children = [child_exp]     
            
        elif func is sympy.core.mul.Mul and (type(children[0]) is sympy.core.numbers.Integer):
            if children[0] < 0:
                func = 'neg'
                child_exp = abs(children[0])*children[1]
                for i in range(2, len(children)):
                    child_exp = child_exp * children[i]
                children = [child_exp]
            else:
                k = int(children[0])
                if k < 4:
                    func = sympy.core.add.Add
                    child_exp = children[1]
                    for i in range(2, len(children)):
                        child_exp = child_exp * children[i]
                    children = [child_exp]*k
                
        if func is sympy.core.power.Pow and children[1] == -1:
            # inversion
            tree_dict[idx] = (expr, 'inv', [])
            tree_dict, c_idx = get_sympy_dag(children[0], tree_dict)
            tree_dict[idx] = (expr, 'inv', [c_idx])
        elif func is sympy.core.power.Pow and children[1] < 0:
            # inversion higher exponent
            tree_dict[idx] = (expr, 'inv', [])
            n = abs(int(children[1]))
            tree_dict, c_idx = get_sympy_dag(children[0]**n, tree_dict)
            tree_dict[idx] = (expr, 'inv', [c_idx])
            
        elif func is sympy.core.power.Pow and children[1] == 0.5:
            # sqrt
            tree_dict[idx] = (expr, 'inv', [])
            n = abs(int(children[1]))
            tree_dict, c_idx = get_sympy_dag(children[0]**n, tree_dict)
            tree_dict[idx] = (expr, 'inv', [c_idx])
        
        else:            
            if func is sympy.core.power.Pow:
                n = int(children[1])
                func = sympy.core.mul.Mul
                children = [children[0]]*n
            
            
            tree_dict[idx] = (expr, func, [])
            if len(children) == 1:
                tree_dict, c_idx = get_sympy_dag(children[0], tree_dict)
                tree_dict[idx] = (expr, func, [c_idx])
            elif len(children) > 1:
                tree_dict[idx] = (expr, func, [])
                new_idxs = len(children) - 2
                for i in range(new_idxs):
                    tree_dict[idx + i + 1] = (expr, func, [])
                child_idxs = []
                for c in children:
                    tree_dict, c_idx = get_sympy_dag(c, tree_dict)
                    child_idxs.append(c_idx)
                
                tree_dict[idx + new_idxs] = (None, func, child_idxs[:2])
                for i in range(len(child_idxs) - 2):
                    j = idx + new_idxs - (i + 1) 
                    tree_dict[j] = (None, func, [j + 1, child_idxs[2+i]])
                _, _, tmp = tree_dict[idx]
                tree_dict[idx] = (expr, func, tmp)
            else:
                # leaf node
                tree_dict[idx] = (expr, func, [])
    
    return tree_dict, idx

def expressions2dag(expr1, expr2):
    cgraph1 = sympy2dag(expr1)
    cgraph2 = sympy2dag(expr2)
    
    node_dict1 = cgraph1.node_dict
    node_dict2 = cgraph2.node_dict

    m1, n1, k1 = cgraph1.inp_dim, cgraph1.outp_dim, cgraph1.n_consts
    m2, n2, k2 = cgraph2.inp_dim, cgraph2.outp_dim, cgraph2.n_consts

    assert m1==m2

    transl_dict1 = {}
    for i in node_dict1:
        if i >= m1 + k1:
            j = i + k2
            if i >= m1 + k1 + n1:
                j += n2
        else:
            j = i
        transl_dict1[i] = j

    transl_dict2 = {}
    for i in node_dict2:
        if i >= m2 and i < (m2 + k2):
            # constant
            j = i + k1
        elif i >= m2 + k2 and i < (m2 + k2 + n2):
            # output
            j = i + k1 + n1

        elif i >= m2 + k2 + n2:
            # others
            j = i + len(node_dict1) - 1
        else:
            # input
            j = i
        transl_dict2[i] = j

    new_dict = {}
    const_count = 0
    for i in node_dict1:
        children, op = node_dict1[i]
        if op.startswith('c_') or op.startswith('const'):
            op = f'c_{const_count}'
            const_count += 1
        new_children = [transl_dict1[j] for j in children]
        new_dict[transl_dict1[i]] = (new_children, op)
    for i in node_dict2:
        children, op = node_dict2[i]
        if op.startswith('c_') or op.startswith('const'):
            op = f'c_{const_count}'
            const_count += 1
        new_children = [transl_dict2[j] for j in children]
        new_dict[transl_dict2[i]] = (new_children, op)
    new_cgraph = reduce_graph(CompGraph(m1, n1+n2, k1+k2, new_dict))
    return new_cgraph

def sympy2dag(expr, m):
    sympy_dict, _ = get_sympy_dag(expr)
    
    
    const_counter = 0
    transl_dict = {
        sympy.core.add.Add : '+',
        sympy.core.mul.Mul : '*',
        sympy.exp : 'exp',
        sympy.log : 'log',
        sympy.cos : 'cos',
        sympy.sin : 'sin'
    }
    new_dict = {}
    const_dict = {}
    seen_inps = []
    for i in sympy_dict:
        expr, func, parents = sympy_dict[i]
        if func in transl_dict:
            new_dict[i] = (transl_dict[func], parents)
        else:
            if func is sympy.core.symbol.Symbol:
                v, c = str(expr).split('_')
                if v == 'x':
                    new_dict[i] = (str(expr), parents)
                    seen_inps.append(int(c.replace('{', '').replace('}', '')))
                else:
                    new_dict[i] = (f'c_{const_counter}', parents)
                    const_dict[const_counter] = None
                    const_counter += 1
            elif expr.is_number:
                new_dict[i] = (f'c_{const_counter}', parents)
                const_dict[const_counter] = float(sympy.re(expr))
                const_counter += 1

            else:
                new_dict[i] = (func, parents)

    if len(new_dict) == 1:
        # ident!
        new_dict = {0 : ('=', [1]),
                    1 : new_dict[0]}

    for i in range(m):
        if i not in seen_inps:
            new_idx = max(new_dict.keys())
            new_dict[new_idx + 1] = (f'x_{i}', [])

    
    outp_idxs = [0]
    inp_idxs = [i for i in new_dict if new_dict[i][0].startswith('x_')]
    const_idxs = [i for i in new_dict if new_dict[i][0].startswith('c_')]
    inp_sort_idxs = np.argsort(np.array([int(new_dict[i][0].split('_')[1].replace('{', '').replace('}', '')) for i in inp_idxs]))
    const_sort_idxs = np.argsort(np.array([int(new_dict[i][0].split('_')[1].replace('{', '').replace('}', '')) for i in const_idxs]))

    inp_idxs = [inp_idxs[i] for i in inp_sort_idxs]
    const_idxs = [const_idxs[i] for i in const_sort_idxs]

    taken_idxs = set(inp_idxs + const_idxs + outp_idxs)
    rest_idxs = [i for i in new_dict if i not in taken_idxs]
    
    transl_dict = {j: i for i, j in enumerate(inp_idxs + const_idxs + outp_idxs + rest_idxs)}

    ret_dict = {}
    for i in new_dict:
        op, parents = new_dict[i]
        ret_dict[transl_dict[i]] = ([transl_dict[j] for j in parents], op)

    m = len(inp_idxs)
    n = 1
    k = len(const_idxs)
    

    cgraph = reduce_graph(CompGraph(m, n, k, ret_dict))

    return cgraph, const_dict

def change_operation(cgraph):
    bin_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 2]
    un_ops = [op for op in config.NODE_ARITY if config.NODE_ARITY[op] == 1]
    
    ret_graph = cgraph.copy()
    
    # 1. select node
    comp_nodes = [i for i in range((cgraph.inp_dim + cgraph.n_consts), len(cgraph.node_dict))]
    node_idx = np.random.choice(comp_nodes)
    
    # 2. change operation
    parents, op = cgraph.node_dict[node_idx]
    if len(parents) == 2:
        new_op = np.random.choice([x for x in bin_ops if x != op])
    else:
        new_op = np.random.choice([x for x in un_ops if x != op])
    
    ret_graph.node_dict[node_idx] = (parents, new_op)
    return ret_graph

def change_edge(cgraph):
    ret_graph = cgraph.copy()

    # 1. select node
    comp_nodes = [i for i in range((cgraph.inp_dim + cgraph.n_consts), len(cgraph.node_dict))]
    node_idx = np.random.choice(comp_nodes)
    
    # 2. select new parent
    parents, op = cgraph.node_dict[node_idx]
    parent_idx = np.random.randint(len(parents))
    p_old = parents[parent_idx]
    possible_parents = [i for i in cgraph.eval_order[:cgraph.eval_order.index(node_idx)] if i != p_old]
    p_new = np.random.choice(possible_parents)
    
    if len(parents) == 1:
        new_parents = [p_new]
    else:
        new_parents = sorted([p_new, parents[int(1-parent_idx)]])
    
    
    ret_graph.node_dict[node_idx] = (new_parents, op)
    return reduce_graph(ret_graph)

def reduce_graph(comp_graph:CompGraph) -> CompGraph:
    '''
    Reduces the computational graph in 2 steps:
    1.  nodes with the same paths leading to them are collapsed into one, 
        identity nodes that are no output are collapsed with their successor
    2.  nodes that have no path to an output node are removed


    @Params:
        comp_graph... computational graph to be reduced

    @Returns:
        reduced graph
    '''
    cgraph = comp_graph.copy()

    nodes1 = [i for i in cgraph.node_dict if i >= cgraph.inp_dim + cgraph.n_consts + cgraph.outp_dim] # if match they are removed
    nodes2 = [i for i in cgraph.node_dict if i >= cgraph.inp_dim + cgraph.n_consts] # if match they take edges from counterpart

    # 1. collapse doubled nodes

    # H is a list of tuples (keep, remove)
    H = []
    for v in nodes1:
        for u in nodes2:
            if u > v:
                l1, op1 = cgraph.node_dict[u]
                l2, op2 = cgraph.node_dict[v]
                if (sorted(l1), op1) == (sorted(l2), op2):
                    H.append((u,v))

    # identity nodes that are no output nodes
    for v in cgraph.node_dict:
        children, op = cgraph.node_dict[v]

        if (op == '=') and (v not in cgraph.outp_nodes):
            u = children[0] # should be the only child
            H.append((u,v))  
            
    if len(H) > 0:
        # 1. resolve H
        # delete v
        transl_dict = {}
        for u, v in H:
            if v in cgraph.node_dict:
                del cgraph.node_dict[v]
            transl_dict[v] = u
        for i in cgraph.node_dict:
            if i not in transl_dict:
                transl_dict[i] = i

        # replace all references to v by u
        tmp = {}
        for i in cgraph.node_dict:
            children, op = cgraph.node_dict[i]
            new_i = transl_dict[i]
            while new_i not in cgraph.node_dict:
                new_i = transl_dict[new_i]
            new_children = []
            for c in children:
                new_c = transl_dict[c]
                while new_c not in cgraph.node_dict:
                    new_c = transl_dict[new_c]
                new_children.append(new_c)
            
            tmp[new_i] = (new_children, op)
        # update indices
        transl_dict = {i : j for j, i in enumerate(sorted(tmp.keys()))}
        new_node_dict = {}
        for i in tmp:
            children, op = tmp[i]
            new_children = [transl_dict[c] for c in children]
            new_node_dict[transl_dict[i]] = (new_children, op)
        cgraph.node_dict = new_node_dict
        cgraph.update_stats() 

    
    # Special case: Remove unneccessary identity nodes at output
    H = {}
    V = set()
    for v in cgraph.node_dict:
        children, op = cgraph.node_dict[v]
        if (op == '='):
            u = children[0] # should be the only child
            if (u not in cgraph.outp_nodes) and (v in cgraph.outp_nodes) and (u not in cgraph.inp_nodes) and (u not in H):
                # delete v, replace all references to u with v
                H[u] = v
                V.add(v)
    tmp = {}
    for i in cgraph.node_dict:
        if i not in V:
            children, op = cgraph.node_dict[i]
            new_children = []
            for c in children:
                if c in H:
                    new_children.append(H[c])
                elif c not in V:
                    new_children.append(c)
            if i in H:
                tmp[H[i]] = new_children, op
            else:
                tmp[i] = new_children, op
    # re-enumerate
    transl_dict = {old_idx : new_idx for new_idx, old_idx in enumerate(sorted(tmp.keys()))}
    new_node_dict = {}
    for i in tmp:
        children, op = tmp[i]

        new_children = []
        for c in children:
            new_children.append(transl_dict[c])
        new_node_dict[transl_dict[i]] = (new_children, op)


    cgraph.node_dict = new_node_dict
    cgraph.update_stats()
    
              
    # 2. remove deadends
    rem_mask = (np.sum((cgraph.R[cgraph.outp_nodes,:] > 0), axis=0) == 0)
    rem_mask[cgraph.inp_nodes] = False

    rem_idx = np.arange(0, len(cgraph.node_dict), 1)[rem_mask]

    tmp = {i : cgraph.node_dict[i] for i in cgraph.node_dict}
    for idx in rem_idx:
        del tmp[idx]  
    transl_dict = {old_idx : new_idx for new_idx, old_idx in enumerate(sorted(tmp.keys()))}
    new_node_dict = {}
    for i in tmp:
        children, op = cgraph.node_dict[i]

        new_children = []
        for c in children:
            new_children.append(transl_dict[c])
        new_node_dict[transl_dict[i]] = (new_children, op)
    cgraph.node_dict = new_node_dict
    cgraph.update_stats()
    return cgraph

def collapse_nodes(cgraph, param_nodes = []):
    # 1. which nodes do depend exclusively on constants?
    const_dependence = {} # on which constants does node depend?
    const_nodes = [] # nodes that depend solely on constants
    c_nodes = set([i + cgraph.inp_dim for i in range(cgraph.n_consts) if i not in param_nodes])
    for node in cgraph.eval_order:
        dep_set = set()
        parents, op = cgraph.node_dict[node]

        isvalid = True
        if ((op == 'const') or (op.startswith('c_'))) and node in c_nodes:
            dep_set.add(node)
        else:
            if len(parents) == 0:
                isvalid = False
            for p in parents:
                p_set = const_dependence[p]
                dep_set = dep_set | p_set
                isvalid = isvalid and (p in const_nodes)
        const_dependence[node] = dep_set
        if isvalid:
            const_nodes.append(node)

    collapse_nodes = []

    possible_nodes = [node for node in const_nodes if len(const_dependence[node]) > 0]
    for node in possible_nodes:
        # which nodes do depend on same constants?
        ref_nodes = [node2 for node2 in cgraph.node_dict if (len(const_dependence[node] & const_dependence[node2]) > 0) and node2 != node]
        ref_nodes = set(ref_nodes)

        # predecessors
        predecessors = set(np.where(cgraph.R[node] > 0)[0])
        predecessors.remove(node)

        successors = set(np.where(cgraph.R[:, node] > 0)[0])
        successors.remove(node)

        # For all nodes that depend on same constants must either hold:
        # - they are predecessors or
        # - they are successors and all paths from constants to successor lead over node
        valid_successors = set()
        for s in successors:
            common_consts = (const_dependence[node] & const_dependence[s])
            if len(common_consts) > 0:
                valid = True
                for c in common_consts:
                    n1 = cgraph.R[s, c]
                    n2 = cgraph.R[s, node]
                    valid = valid and (n1==n2)
                if valid:
                    valid_successors.add(s)
        
        if (len(ref_nodes & (predecessors | valid_successors)) == len(ref_nodes)) and len(ref_nodes) > 0:
            collapse_nodes.append(node)

    final_collapse_nodes = []
    for node in collapse_nodes:
        # is any node in collapse nodes a successor? is node a constant already?, is node an output node?-> ignore
        is_const = node in c_nodes
        is_outp = node in cgraph.outp_nodes
        sucessors = set(np.where(cgraph.R[:, node] > 0)[0])
        sucessors.remove(node)

        if (not is_const) and (not is_outp) and (len(successors & set(collapse_nodes)) == 0):
            final_collapse_nodes.append(node)

    # nodes to be deleted: idle nodes (except input) + predecessors of collapse nodes
    del_consts = set()
    del_nodes = set()
    for node in final_collapse_nodes:
        del_consts = del_consts | const_dependence[node]
        predecessors = set(np.where(cgraph.R[node] > 0)[0])
        predecessors.remove(node)
        del_nodes = del_nodes | predecessors
    mask = np.ones(len(cgraph.node_dict)).astype(bool)
    for i in range(cgraph.outp_dim):
        mask = mask & (cgraph.R[cgraph.inp_dim + cgraph.n_consts + i] == 0)
    for node in np.where(mask)[0]:
        if node in c_nodes:
            del_consts.add(node)
            
        if node >= cgraph.inp_dim:
            del_nodes.add(node)
    final_collapse_nodes = [node for node in final_collapse_nodes if node not in del_nodes]


    new_k = cgraph.n_consts + len(final_collapse_nodes) - len(del_consts)

    # add collapse nodes as new constants

    transl_dict = {}
    counter = 0
    for node in cgraph.node_dict:
        if node < cgraph.inp_dim + cgraph.n_consts:
            transl_dict[node] = node
        elif node in final_collapse_nodes:
            transl_dict[node] = cgraph.inp_dim + cgraph.n_consts + counter
            counter += 1
        elif node not in del_nodes:
            transl_dict[node] = node + len(final_collapse_nodes)

    new_node_dict = {}
    for node in cgraph.node_dict:
        if node not in del_nodes:
            if node in final_collapse_nodes:
                new_node_dict[transl_dict[node]] = [], 'const'
            else:
                parents, op = cgraph.node_dict[node]
                new_parents = [transl_dict[p] for p in parents]
                new_node_dict[transl_dict[node]] = new_parents, op

    transl_dict = {}        
    for i, node in enumerate(sorted(new_node_dict)):
        transl_dict[node] = i

    new_node_dict_reduced = {}
    for node in sorted(new_node_dict):
        parents, op = new_node_dict[node]
        new_parents = [transl_dict[p] for p in parents]
        new_node_dict_reduced[transl_dict[node]] = new_parents, op
    new_cgraph = CompGraph(cgraph.inp_dim, cgraph.outp_dim, new_k, new_node_dict_reduced)
    return new_cgraph

def plot_cgraph(cgraph:CompGraph, ax=plt.axes, radius = 0.1, buffer_factor = 5.0, curvature = 0.3) -> nx.MultiDiGraph:
    '''
    Tries to plot the given computational graph.

    THIS IS NOT GUARANTEED TO GIVE NICE RESULTS!

    @Params:
        cgraph... computational graph to be plotted
        ax... matplotlib axes to plot on
        radius... determines node size (0.1 seems to work quite well)
        buffer_factor... determines spacing of plots around nodes
        curvature... if multiple edges from source to target, they are bent with this curvature


    @Returns:
        networkx Graph
    '''


    # get layers of nodes
    eval_order = cgraph.get_eval_order()
    
    # get positions for nodes

    y_max = max([len(order) for order in eval_order])
    y_min = 0

    node_positions = {}
    for x, node_list in enumerate(eval_order):
        y_pos = np.flip(np.linspace(y_min, y_max, len(node_list) + 2)[1:-1])
        for n, y in zip(node_list, y_pos):
            node_positions[n] = (x, y)


    # create Graph

    G = nx.MultiDiGraph()

    # add nodes
    color_dict = {}
    label_dict = {}

    inp_count = 0
    const_count = 0
    for i in cgraph.node_dict:
        children, op = cgraph.node_dict[i]
        G.add_node(i)

        # color
        if i in cgraph.inp_nodes:
            color_dict[i] = 'cyan'
        elif i in cgraph.outp_nodes:
            color_dict[i] = 'red'
        else:
            color_dict[i] = 'white'


        # label
        if op == 'inp':
            op = f'x{inp_count}'
            inp_count += 1
        elif op == 'const':
            op = f'c{const_count}'
            const_count += 1
        label_dict[i] = op


    # add edges
    for i in cgraph.node_dict:
        children, op = cgraph.node_dict[i]
        for j in children:
            G.add_edge(j,i)

    color_list = [color_dict[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, node_positions, node_color=color_list, edgecolors='black', node_size = (2*radius)*10**4, ax = ax)
    nx.draw_networkx_labels(G, node_positions, labels=label_dict, ax = ax)
    
    for e in G.edges:
        
        
        x0, y0 = node_positions[e[0]]
        x1, y1 = node_positions[e[1]]
        
        sign = (e[2]%2)*2-1
        ampl = (((e[2]+1)//2))*curvature
        ax.annotate("",
                    xy=(x0, y0), xycoords='data',
                    xytext=(x1, y1), textcoords='data',
                    arrowprops=dict(arrowstyle="<|-", color="0.0",
                                    shrinkA=100*radius, shrinkB=100*radius,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr',str(sign*ampl)
                                    ),
                                    ),
                    )
    
    

    buffer = buffer_factor*radius
    xs = []
    ys = []
    for n in node_positions:
        (x,y) = node_positions[n]
        xs.append(x)
        ys.append(y)
    
    ymin, ymax = np.min(ys), np.max(ys)
    xmin, xmax = np.min(xs), np.max(xs)
    ax.set_ylim(ymin-buffer, ymax+buffer)
    ax.set_xlim(xmin-buffer, xmax+buffer)
    
    return G



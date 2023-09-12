import numpy as np
import torch
import sympy

'''
Constants
'''
CONST_POOL = [1, 2, 10, np.pi, np.e]

'''
Numeric Operations
'''
NODE_OPS = {
    '+' : lambda x, y: x + y,
    '*' : lambda x, y: x * y,
    'sub_l' : lambda x, y: x - y,
    'sub_r' : lambda x, y: y - x,
    'div_l' : lambda x, y: x/y,
    'div_r' : lambda x, y: y/x,
    #'pow_l' : lambda x, y: x**y,
    #'pow_r' : lambda x, y: y**x,
    '=' : lambda x: x,
    'inv' : lambda x : 1/x,
    'neg' : lambda x : -x,
    'sin' : np.sin,
    'cos' : np.cos,
    'exp' : np.exp,
    'log' : np.log,
    'sqrt' : np.sqrt,
    'sq' : lambda x: x**2
}

NODE_OPS_PYTORCH = {
    '+' : lambda x, y: x + y,
    '*' : lambda x, y: x * y,
    'sub_l' : lambda x, y: x - y,
    'sub_r' : lambda x, y: y - x,
    'div_l' : lambda x, y: x/y,
    'div_r' : lambda x, y: y/x,
    #'pow_l' : lambda x, y: x**y,
    #'pow_r' : lambda x, y: y**x,
    '=' : lambda x: x,
    'inv' : lambda x : 1/x,
    'neg' : lambda x : -x,
    'sin' : torch.sin,
    'cos' : torch.cos,
    'exp' : torch.exp,
    'log' : torch.log,
    'sqrt' : torch.sqrt,
    'sq' : lambda x: x**2
}

NODE_ARITY = {
    '+' : 2,
    '*' : 2,
    'sub_l' : 2,
    'sub_r' : 2,
    'div_l' : 2,
    'div_r' : 2,
    #'pow_l' : 2,
    #'pow_r' : 2,
    '=' : 1,
    'inv' : 1,
    'neg' : 1,
    'sin' : 1, 
    'cos' : 1,
    'exp' : 1,
    'log' : 1,
    'sqrt' : 1,
    'sq' : 1
}

NODE_STR = {
    '+' : '(a)+(b)',
    '*' : '(a)*(b)',
    'sub_l' : '(a)-(b)',
    'sub_r' : '(b)-(a)',
    'div_l' : '(a)/(b)',
    'div_r' : '(b)/(a)',
    #'pow_l' : '(a)**(b)',
    #'pow_r' : '(b)**(a)',
    '=' : '(a)',
    'inv' : '1/(a)',
    'neg' : '-(a)',
    'sin' : 'np.sin(a)',
    'cos' : 'np.cos(a)',
    'exp' : 'np.exp(a)',
    'log' : 'np.log(a)',
    'sqrt' : 'np.sqrt(a)',
    'sq' : '(a)**2'
}

'''
Symbolic Operations
'''

NODE_OPS_SYMB = {
    '+' : lambda x, y: x + y,
    '*' : lambda x, y: x * y,
    'sub_l' : lambda x, y: x - y,
    'sub_r' : lambda x, y: y - x,
    'div_l' : lambda x, y: x/y,
    'div_r' : lambda x, y: y/x,
    #'pow_l' : lambda x, y: x**y,
    #'pow_r' : lambda x, y: y**x,
    '=' : lambda x: x,
    'inv' : lambda x : 1/x,
    'neg' : lambda x : -x,
    'sin' : lambda x: sympy.sin(x),
    'cos' : lambda x: sympy.cos(x),
    'exp' : lambda x: sympy.exp(x),
    'log' : lambda x: sympy.log(x),
    'sqrt' : lambda x: sympy.sqrt(x),
    'sq' : lambda x: x**2
}

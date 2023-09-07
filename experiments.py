import numpy as np
import pickle
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics import r2_score
import os
import regressors
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

from DAG_search import utils
from DAG_search import dag_search

import networkx as nx

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

def recovery_experiment(ds_name : str, regressor, regressor_name : str, is_symb : bool, test_size : float = 0.2):
    '''
    Simple Experiment to estimate the Recovery rate of a Regressor.

    @Params:
        ds_name... Name of dataset
        regressor... Scikit learn style regressor (with .fit(X, y) and .predict(X))
        regressor_name... name of regressor (for saving)
        is_symb... if regressor is symbolic
        test_size... share of test data
    '''
    
    load_path = f'datasets/{ds_name}/tasks.p'
    save_path = f'results/{ds_name}/{regressor_name}_results.p'

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')

    results = {}
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    problems = list(task_dict.keys())

    for problem in problems:
        print('####################')
        print(f'{regressor_name} on {problem}')
        print('####################')

        X, y, exprs_true = task_dict[problem]['X'], task_dict[problem]['y'], task_dict[problem]['expr']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        all_rec = []
        all_pred_train = []
        all_pred_test = []
        all_y_train = []
        all_y_test = []
        all_expr = []
        all_times = []


        for idx in range(y.shape[1]):
            expr_true = exprs_true[idx]
            y_part = y_train[:, idx]

            s_time = timer()
            regressor.fit(X_train, y_part)
            e_time = timer()
            all_times.append(e_time - s_time)

            pred = regressor.predict(X_train)
            all_pred_train.append(pred)
            all_y_train.append(y_part)

            pred = regressor.predict(X_test)
            all_pred_test.append(pred)
            all_y_test.append(y_test[:, idx])



            if is_symb:
                expr_est = regressor.model()
                all_expr.append(expr_est)
                rec = utils.symb_eq(expr_est, expr_true) 
            else:
                rec = False
            all_rec.append(rec)

    
        results[problem] = {
            'recovery' : all_rec,
            'exprs' : all_expr,
            'y_train' : all_y_train,
            'y_test' : all_y_test,
            'pred_train' : all_pred_train,
            'pred_test' : all_pred_test,
            'times' : all_times
        }

        with open(save_path, 'wb') as handle:
            pickle.dump(results, handle)

def proximity_experiment(random_state = None):
    # sample graphs
    save_path_dists = f'results/distance_dict.p'
    save_path_corr = f'results/corr_matrix.npy'

    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)

    m = 1
    n_graphs = 100
    X = np.random.rand(100, m)
    y_target = X[:, 0]
    graphs = []
    while len(graphs) < n_graphs:
        graph = dag_search.sample_graph(m = m, n = 1, k = 0, n_calc_nodes = 5)
        pred = graph.evaluate(X, c = np.array([]))
        # make sure its valid on input data
        valid = not (np.any(np.isnan(pred)) or np.any(np.isinf(pred)))
        if valid:
            r2 = r2_score(pred[:, 0], y_target) # so R2 is in [0, 1]
            if r2 > 0.0:
                graphs.append(graph)
    

    # Numeric distances
    print('Numeric distances')
    MSE_mat = np.zeros((len(graphs), len(graphs)))
    RMSE_mat = np.zeros((len(graphs), len(graphs)))
    MAE_mat = np.zeros((len(graphs), len(graphs)))

    for i, graph1 in tqdm(enumerate(graphs), total = len(graphs)):
        for j, graph2 in enumerate(graphs):
            if i <= j:
                pred1 = graph1.evaluate(X, c = np.array([]))
                pred2 = graph2.evaluate(X, c = np.array([]))

                mse = np.mean((pred1 - pred2)**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(pred1 - pred2))

                MSE_mat[i, j] = mse
                MSE_mat[j, i] = mse
                
                RMSE_mat[i, j] = rmse
                RMSE_mat[j, i] = rmse
                
                MAE_mat[i, j] = mae
                MAE_mat[j, i] = mae

    print('R2 Score')
    R2_mat = np.zeros((len(graphs), len(graphs)))
    for i, graph1 in tqdm(enumerate(graphs), total = len(graphs)):
        for j, graph2 in enumerate(graphs):
            pred1 = graph1.evaluate(X, c = np.array([]))
            pred2 = graph2.evaluate(X, c = np.array([]))
            r2 = r2_score(pred1[:, 0], pred2[:, 0])
            R2_mat[i, j] = r2

    print('DFS-order')
    dfs_mat = np.zeros((len(graphs), len(graphs)))
    for i, graph1 in tqdm(enumerate(graphs), total = len(graphs)):
        for j, graph2 in enumerate(graphs):
            if i <= j:
                r = utils.dfs_ratio(graph1, graph2)
                dfs_mat[i, j] = r
                dfs_mat[j, i] = r

    print('Depth of Subexpressions')
    subexp_mat = np.zeros((len(graphs), len(graphs)))
    for i, graph1 in tqdm(enumerate(graphs), total = len(graphs)):
        for j, graph2 in enumerate(graphs):
            if i <= j:
                r = utils.subexpr_ratio(graph1, graph2)
                subexp_mat[i, j] = r
                subexp_mat[j, i] = r

    print('Edit distance')
    edit_mat = np.zeros((len(graphs), len(graphs)))
    for i, graph1 in tqdm(enumerate(graphs), total = len(graphs)):
        for j, graph2 in enumerate(graphs):
            if i <= j:
                r = utils.edit_distance(graph1, graph2)
                edit_mat[i, j] = r
                edit_mat[j, i] = r

    dist_dict = {
        'MSE' : MSE_mat.flatten(),
        'RMSE' : RMSE_mat.flatten(),
        'MAE' : MAE_mat.flatten(),
        'R2' : R2_mat.flatten(),
        'edit' : edit_mat.flatten(),
        'preorder' : dfs_mat.flatten(),
        'subexpr' : subexp_mat.flatten(),
    }

    corr_matrix = np.zeros((len(dist_dict), len(dist_dict)))
    methods = list(dist_dict.keys())
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            v1 = dist_dict[method1]
            v2 = dist_dict[method2]

            corr, _ = pearsonr(v1, v2) # small p value = significant
            corr_matrix[i, j] = corr
    
    with open(save_path_dists, 'wb') as handle:
        pickle.dump(dist_dict, handle)
    np.save(save_path_corr, corr_matrix)

def solutions_experiment(topk : int = 10, k : int = 1, n_graphs : int = 10000, random_state : int = None):
    '''
    Experiment to evaluate the symbolic similarity of top performers.

    @Params:
        topk... top performers
        k... number of constants
        n_graphs... size of population
    '''
    
    save_path = 'results/local_minima_exp.p'
    if not os.path.exists('results'):
        os.mkdir(save_path)

    if random_state is None:
        np.random.seed(0)
    else:
        np.random.seed(random_state)

    # 1. get different regression tasks
    print('loading tasks')
    load_path = f'datasets/Strogatz/tasks.p'
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
        
    task_list = []
    for problem_name in task_dict:
        X = task_dict[problem_name]['X']
        y_all = task_dict[problem_name]['y']
        for i in range(y_all.shape[1]):
            task_list.append((X, y_all[:, i]))
            
    m = task_list[0][0].shape[1]
    for X, y in task_list:
        assert X.shape[1] == m
    print(f'{len(task_list)} tasks loaded')
    res = {}
    for n_calc_nodes in np.arange(1, 10, 1):
        print('#################')
        print(f'# calc nodes: {n_calc_nodes}')
        print('#################')
        print('sampling population')
        graphs = []
        for _ in tqdm(range(n_graphs)):
            graph = dag_search.sample_graph(m, 1, k, n_calc_nodes)
            graphs.append(graph)
        
        
        ratios = []
        rmses = []
        for problem_idx, (X, y) in enumerate(task_list):
            print(f'\nProblem Nr. {problem_idx + 1}\n')
            
            loss_fkt = dag_search.MSE_loss_fkt(y)

            # 2. sample + optimize population of graphs
            print('optimizing population')
            losses = []
            consts = []
            for graph in tqdm(graphs):
                c, loss = dag_search.evaluate_cgraph(graph, X, loss_fkt, opt_mode = 'grid_zoom')
                losses.append(loss)
                consts.append(c)
                
            # 3. get top performers
            sort_idx = np.argsort(np.array(losses))[:10]
            top_graphs = [graphs[i] for i in sort_idx]
            top_losses = [losses[i] for i in sort_idx]
            top_consts = [consts[i] for i in sort_idx]

            # build similarity matrix
            print('Building similarity matrix')
            eq_mat = np.zeros((len(top_graphs), len(top_graphs)))
            for i, graph1 in tqdm(enumerate(top_graphs), total = len(top_graphs)):
                for j, graph2 in enumerate(top_graphs):
                    if i <= j:
                        expr1 = graph1.evaluate_symbolic(c = top_consts[i])[0]
                        expr2 = graph2.evaluate_symbolic(c = top_consts[j])[0]
                        r = utils.symb_eq(expr1, expr2)
                        #r = utils.edit_distance(graph1, graph2)
                        eq_mat[i, j] = r
                        eq_mat[j, i] = r

            A = eq_mat.astype(int)
            
            n_comps = get_components(A)
            ratios.append(n_comps/topk)
            print(f'Detected solutions: {n_comps}')
            
            # Estimate RMSES
            top_rmses = []
            for graph, c in zip(top_graphs, top_consts):
                pred = graph.evaluate(X, c = c)
                rmse = np.sqrt(np.mean((pred - y)**2))
                top_rmses.append(rmse)
            rmses.append((np.min(top_rmses), np.max(top_rmses), np.mean(top_rmses)))
            print(f'RMSE stats: {rmses[-1]}')
        res[n_calc_nodes] = {
            'group-ratio' : ratios,
            'min-RMSE' : [x[0] for x in rmses],
            'max-RMSE' : [x[1] for x in rmses],
            'avg-RMSE' : [x[2] for x in rmses],
        }
        
        with open(save_path, 'wb') as handle:
            pickle.dump(res, handle)


if __name__ == '__main__':

    # local minima experiment
    if True:
        solutions_experiment(k = 2, n_graphs=10000, random_state=0)

    # recovery experiment
    if False:
        rand_state = 0
        problems = [n for n in os.listdir('datasets') if 'ipynb' not in n]
        regs = {
            'linreg' : (regressors.LinReg(), True),
            'polyreg2' : (regressors.PolyReg(degree= 2), True),
            'polyreg3' : (regressors.PolyReg(degree= 3), True),
            'MLP' : (regressors.MLP(random_state = rand_state), False),
            'operon' : (regressors.Operon(random_state = rand_state), True),
            'gplearn' : (regressors.GPlearn(random_state = rand_state), True),
            'DAGSearch' : (dag_search.DAGRegressor(processes = 10, random_state = rand_state), True),
            #'dsr' : (regressors.DSR(), True),
        }
        for ds_name in problems:
            for regressor_name in regs:
                regressor, is_symb = regs[regressor_name]
                recovery_experiment(ds_name = ds_name, regressor = regressor, regressor_name = regressor_name, is_symb = is_symb)

    # proximity experiment
    if False:
        proximity_experiment(0)
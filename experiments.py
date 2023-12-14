import numpy as np
import pickle
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics import r2_score
import os
import sympy


from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

from DAG_search import utils
from DAG_search import dag_search
from regressors import regressors as sregs


def recovery_experiment(ds_name : str, regressor, regressor_name : str, is_symb : bool, test_size : float = 0.2, overwrite:bool = False):
    '''
    Simple Experiment to estimate the Recovery rate of a Regressor.

    @Params:
        ds_name... Name of dataset
        regressor... Scikit learn style regressor (with .fit(X, y) and .predict(X))
        regressor_name... name of regressor (for saving)
        is_symb... if regressor is symbolic
        test_size... share of test data
    
    @Returns:
        saves dictionary:
            [problem][criterium] = list of results for each dimension
    '''
    
    load_path = f'datasets/{ds_name}/tasks.p'
    save_path = f'results/{ds_name}/{regressor_name}_results.p'

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')

    if os.path.exists(save_path):
        with open(save_path, 'rb') as handle:
            results = pickle.load(handle)
    else:    
        results = {}
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    problems = list(task_dict.keys())

    for problem in problems:

        print('####################')
        print(f'{regressor_name} on {problem}')
        print('####################')
        
        if (problem not in results) or overwrite:

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
                regressor.fit(X_train, y_part, verbose = 2)
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

                print(f'Recovery: {rec}')

        
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

def scaling_experiment(ds_name : str, n_tries : int = 5, inter_nodes : list = [1, 2, 3, 4], orders :list = [10000, 50000, 100000, 200000], test_size : float = 0.2):
    '''
    Experiment to show that more compute = more recovery.
    The two complexity parameters #internal nodes and #DAG frames are varied.

    @Params:
        ds_name... Name of dataset
        n_tries... number of tries (different random states)
        inter_nodes... list of number of internal nodes to try
        orders... list of number of maximum DAG-frames to try
        test_size... test share for testing

    @Returns:
        saves dictionary:
            [rand_state][n_calc_nodes][max_orders][problem][criterium] = list of results for subproblem
    '''

    load_path = f'datasets/{ds_name}/tasks.p'
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    save_path = f'results/{ds_name}/scalings.p'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')
    
    if os.path.exists(save_path):
        with open(save_path, 'rb') as handle:
            res_dict = pickle.load(handle)
    else:
        res_dict = {}

    for rand_state in range(n_tries):
        if rand_state not in res_dict:
            res_dict[rand_state] = {}
        for n_calc_nodes in inter_nodes:
            if n_calc_nodes not in res_dict[rand_state]:
                res_dict[rand_state][n_calc_nodes] = {}
            for max_orders in orders:
                if max_orders not in res_dict[rand_state][n_calc_nodes]:
                    res_dict[rand_state][n_calc_nodes][max_orders] = {}

                regressor = dag_search.DAGRegressor(processes = 32, random_state = rand_state, n_calc_nodes = n_calc_nodes, max_orders = max_orders)
                for problem in task_dict:
                    if problem not in res_dict[rand_state][n_calc_nodes][max_orders]:
                        res_dict[rand_state][n_calc_nodes][max_orders][problem] = {}
                        print('####################')
                        print(f'# Random State: {rand_state}, Nodes: {n_calc_nodes}, Orders: {max_orders}, Problem: {problem}')
                        print('####################')

                        X, y, exprs_true = task_dict[problem]['X'], task_dict[problem]['y'], task_dict[problem]['expr']

                        all_rec = []
                        all_expr = []
                        all_times = []


                        for idx in range(y.shape[1]):
                            expr_true = exprs_true[idx]
                            y_part = y[:, idx]

                            s_time = timer()
                            regressor.fit(X, y_part)
                            e_time = timer()
                            all_times.append(e_time - s_time)

                            expr_est = regressor.model()
                            all_expr.append(expr_est)
                            rec = utils.symb_eq(expr_est, expr_true) 
                            all_rec.append(rec)

        
                        res_dict[rand_state][n_calc_nodes][max_orders][problem] = {
                            'recovery' : all_rec,
                            'exprs' : all_expr,
                            'times' : all_times
                        }   
                        with open(save_path, 'wb') as handle:
                            pickle.dump(res_dict, handle) 

def timing_experiment(ds_name : str, n_cores : list = [1, 2, 4, 8, 16, 32], overwrite:bool = True):
    '''
    Simple Experiment to show parallelizability of DAGSearch on a Regression Problem.

    @Params:
        ds_name... Name of dataset
        n_cores... number of cores to use
    
    @Returns:
        saves dictionary:
            [problem][n_cores] = time
    '''
    load_path = f'datasets/{ds_name}/tasks.p'
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    save_path = f'results/{ds_name}/timings.p'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')
    
    if os.path.exists(save_path):
        with open(save_path, 'rb') as handle:
            res_dict = pickle.load(handle)
    else:
        res_dict = {}
    
    for problem in task_dict:
        if problem not in res_dict:
            res_dict[problem] = {}
        for n_processes in n_cores:
            regressor = dag_search.DAGRegressor(processes = n_processes, random_state = 0, n_calc_nodes = 2, max_orders = int(1e4))

            if (n_processes not in res_dict[problem]) or overwrite:
                # do experiment
                print('####################')
                print(f'# Problem: {problem} Cores: {n_processes}')
                print('####################')
                X, y, exprs_true = task_dict[problem]['X'], task_dict[problem]['y'], task_dict[problem]['expr']
                all_times = []
                for idx in range(y.shape[1]):
                    expr_true = exprs_true[idx]
                    y_part = y[:, idx]

                    s_time = timer()
                    regressor.fit(X, y_part)
                    e_time = timer()
                    all_times.append(e_time - s_time)
                res_dict[problem][n_processes] = all_times
                with open(save_path, 'wb') as handle:
                    pickle.dump(res_dict, handle) 

###############################
# OLD - not used in the paper
###############################

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
                r = utils.graph_edit_distance(graph1, graph2)
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

def solutions_experiment(topk : int = 100, n_calc_nodes : int = 5, k : int = 1, n_graphs : int = 100000, random_state : int = None):
    '''
    Experiment to evaluate the symbolic similarity of top performers.

    @Params:
        topk... top performers
        k... number of constants
        n_graphs... size of population
    '''
    
    save_path = 'results/local_minima_exp.p'
    if not os.path.exists('results'):
        os.mkdir('results')

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
    task_names = []
    for problem_name in task_dict:
        X = task_dict[problem_name]['X']
        y_all = task_dict[problem_name]['y']
        for i in range(y_all.shape[1]):
            task_list.append((X, y_all[:, i]))
            task_names.append(f'{problem_name}_{i}')
            
    m = task_list[0][0].shape[1]
    for X, y in task_list:
        assert X.shape[1] == m
    print(f'{len(task_list)} tasks loaded')

    res = {}
    print('sampling population')
    graphs = []
    for _ in tqdm(range(n_graphs)):
        graph = dag_search.sample_graph(m, 1, k, n_calc_nodes)
        graphs.append(graph)
        
        
    for problem_idx, (X, y) in enumerate(task_list):
        problem_name = task_names[problem_idx]
        print(f'\nProblem {problem_name}\n')  
        loss_fkt = dag_search.MSE_loss_fkt(y)

        # 2. optimize population of graphs
        print('optimizing population')
        losses = []
        consts = []
        for graph in tqdm(graphs):
            c, loss = dag_search.evaluate_cgraph(graph, X, loss_fkt, opt_mode = 'grid_zoom')
            losses.append(loss)
            consts.append(c)
                
        # 3. get top performers
        sort_idx = np.argsort(np.array(losses))[:topk]
        top_graphs = [graphs[i] for i in sort_idx]
        top_losses = [losses[i] for i in sort_idx]
        top_consts = [consts[i] for i in sort_idx]

        # build similarity matrix
        print('Building similarity matrix')
        eq_mat = np.zeros((len(top_graphs), len(top_graphs)))
        for i, graph1 in tqdm(enumerate(top_graphs), total = len(top_graphs)):
            for j, graph2 in enumerate(top_graphs):
                if i <= j:

                    if False:
                        expr1 = graph1.evaluate_symbolic(c = top_consts[i])[0]
                        expr2 = graph2.evaluate_symbolic(c = top_consts[j])[0]
                        r = utils.symb_eq(expr1, expr2)
                    if True:
                        r = int(utils.edit_distance(graph1, graph2) == 1)
                    #r = utils.edit_distance(graph1, graph2)
                    eq_mat[i, j] = r
                    eq_mat[j, i] = r

        A = eq_mat.astype(int)
            
        n_groups = utils.get_components(A)
        print(f'Detected groups: {n_groups}/{topk}')
            
        # Estimate RMSES
        top_rmses = []
        for graph, c in zip(top_graphs, top_consts):
            pred = graph.evaluate(X, c = c)
            rmse = np.sqrt(np.mean((pred - y)**2))
            top_rmses.append(rmse)

        res[problem_name] = {
            'topk' : topk,
            'groups' : n_groups,
            'min-RMSE' : np.min(top_rmses),
            'max-RMSE' : np.max(top_rmses),
            'avg-RMSE' : np.mean(top_rmses),
        }
        
        with open(save_path, 'wb') as handle:
            pickle.dump(res, handle)

def fdc_experiment(ds_name : str, n_graphs :int = 10000, max_tries : int = 10, random_state : int = None):
    if random_state is not None:
        np.random.seed(random_state)

    load_path = f'datasets/{ds_name}/tasks.p'
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    save_path = f'results/{ds_name}/fdc.p'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')
    res_dict = {}
    
    # 1. get Problems
    task_X = []
    task_y = []
    task_exprs = []
    task_problems = []
    for problem in task_dict:
        X = task_dict[problem]['X']
        y = task_dict[problem]['y']
        exprs = task_dict[problem]['expr']
        for i in range(y.shape[1]):
            task_X.append(X)
            task_y.append(y[:, i])
            task_exprs.append(exprs[i])
            task_problems.append(f'{problem}_{i}')


    for idx in range(len(task_problems)):
        X, y, expr_true = task_X[idx], task_y[idx], task_exprs[idx]
        problem = task_problems[idx]
        loss_fkt = dag_search.MSE_loss_fkt(y)
        
        
        # 2. sample population
        
        graphs = []
        consts = []
        losses = []
        exprs = []
        try_counter = 0
        while (len(graphs) < n_graphs) and (try_counter < max_tries): 
            graph = dag_search.sample_graph(m = X.shape[1], n = 1, k = 1, n_calc_nodes = 3)
            c, loss = dag_search.evaluate_cgraph(graph, X, loss_fkt, opt_mode = 'grid_zoom')
            if loss < 1000:
                losses.append(loss)
                graphs.append(graph)
                consts.append(c)
                exprs.append(graph.evaluate_symbolic(c = c)[0])
                try_counter = 0
            else:
                try_counter += 1
                
        # 3. get distances to ground truth
        dists = []
        for expr in exprs:
            dist = utils.expr_edit_distance(expr, expr_true)
            dists.append(dist)
            
        # 4. Correlation coefficient (only if sufficient)
        if len(dists) > 10:
            r, p_value = pearsonr(losses, dists)
            res_dict[task_problems[idx]] = r
            print(f'Problem {task_problems[idx]}, CDF: {r}')

    # save
    with open(save_path, 'wb') as handle:
        pickle.dump(res_dict, handle)

def local_minima_experiment(ds_name : str, n_graphs :int = 1000, max_tries : int = 10, random_state : int = None):
    if random_state is not None:
        np.random.seed(random_state)

    load_path = f'datasets/{ds_name}/tasks.p'
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    save_path = f'results/{ds_name}/local_minima.p'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')
    res_dict = {}
    
    # 1. get Problems
    task_X = []
    task_y = []
    task_exprs = []
    task_problems = []
    for problem in task_dict:
        X = task_dict[problem]['X']
        y = task_dict[problem]['y']
        exprs = task_dict[problem]['expr']
        for i in range(y.shape[1]):
            task_X.append(X)
            task_y.append(y[:, i])
            task_exprs.append(exprs[i])
            task_problems.append(f'{problem}_{i}')


    for idx in range(len(task_problems)):
        X, y, expr_true = task_X[idx], task_y[idx], task_exprs[idx]
        problem = task_problems[idx]
        loss_fkt = dag_search.MSE_loss_fkt(y)
        
        
        # 2. sample population
        
        graphs = []
        consts = []
        losses = []
        exprs = []
        try_counter = 0
        while (len(graphs) < n_graphs) and (try_counter < max_tries): 
            graph = dag_search.sample_graph(m = X.shape[1], n = 1, k = 1, n_calc_nodes = 3)
            c, loss = dag_search.evaluate_cgraph(graph, X, loss_fkt, opt_mode = 'grid_zoom')
            if loss < 1000:
                losses.append(loss)
                graphs.append(graph)
                consts.append(c)
                exprs.append(graph.evaluate_symbolic(c = c)[0])
                try_counter = 0
            else:
                try_counter += 1
        losses = np.array(losses)
                
        if len(exprs) > 10:       
            # 3. collect local minima
            # check: does expression have connection to anyone better? -> no local minimum else: local minimum
            sort_idx = np.argsort(losses)
            minima = np.ones(len(losses)).astype(bool)
            for i in tqdm(range(len(losses))):
                idx1 = sort_idx[i]
                expr1 = exprs[idx1]
                for j in range(i):
                    idx2 = sort_idx[j]
                    expr2 = exprs[idx2]
                    r = utils.expr_edit_distance(expr1, expr2)
                    if r == 1:
                        minima[idx1] = False
                        break
                
            ratio = np.sum(minima)/len(minima)
            print(problem, ratio)

            res_dict[task_problems[idx]] = ratio
            
            # save
            with open(save_path, 'wb') as handle:
                pickle.dump(res_dict, handle)

def covariance_experiment(ds_name : str, max_tries : int = 10, n_graphs : int = 10000):
    np.random.seed(0)
    # Problem
    load_path = f'datasets/{ds_name}/tasks.p'
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    save_path = f'results/{ds_name}/covariances.p'
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(f'results/{ds_name}'):
        os.mkdir(f'results/{ds_name}')


    problems = list(task_dict.keys())
    results_dict = {}
    for problem in problems:
        X = task_dict[problem]['X']
        y = task_dict[problem]['y']
        exprs = task_dict[problem]['expr']
        
        for part_idx in range(y.shape[1]):
            print('####################')
            print(f'{problem}_{part_idx}')
            print('####################')
            
            y_part = y[:, part_idx]
            target_expr = exprs[part_idx]
            
            
            # get traits of target
            target_traits = utils.get_subexprs_sympy(target_expr)[1:]
            sort_idx = np.argsort([-utils.tree_size(expr) for expr in target_traits])
            target_traits = [target_traits[i] for i in sort_idx]
            target_traits = list(set([str(expr) for expr in target_traits]))
            
            
            # sample population
            loss_fkt = dag_search.MSE_loss_fkt(y_part.reshape(-1, 1))

            population = []
            losses = []
            try_counter = 0
            while (len(population) < n_graphs) and (try_counter < max_tries): 
                graph = dag_search.sample_graph(m = X.shape[1], n = 1, k = 1, n_calc_nodes = 3)
                c, loss = dag_search.evaluate_cgraph(graph, X, loss_fkt, opt_mode = 'grid_zoom')
                if loss < 1000:
                    population.append(graph.evaluate_symbolic(c = c)[0])
                    losses.append(loss)
                    try_counter = 0
                else:
                    try_counter += 1

            if len(population) > 2:  
                fitnesses = -np.array(losses)

                sort_idx = np.argsort(fitnesses)
                population = [population[i] for i in sort_idx]
                fitnesses = fitnesses[sort_idx]

                all_idxs = np.arange(len(population))
                subset_idxs = []
                for i in range(500): 
                    np.random.shuffle(all_idxs)
                    sub_idxs = all_idxs[:100].copy()
                    subset_idxs.append(sub_idxs)
                subset_idxs = np.row_stack(subset_idxs)


                covs = []
                all_scores = []
                all_occs = []
                for trait in tqdm(target_traits):
                    scores = []
                    occurences = []
                    for sub_idxs in subset_idxs:
                        sub_population = [population[i] for i in sub_idxs]
                        sub_fitnesses = np.array(fitnesses)[sub_idxs]
                        scores.append(np.mean(sub_fitnesses))
                        occurences.append(utils.expr_occurence(trait, sub_population))

                    covariance = np.cov(np.row_stack([scores, occurences]))[0, 1]
                    covs.append(covariance)
                    all_scores.append(np.array(scores))
                    all_occs.append(np.array(occurences))
                    
                
                results_dict[f'{problem}_{part_idx}'] = {
                    'covariances' : covs,
                    'scores' : all_scores,
                    'occurences' : all_occs
                }

                with open(save_path, 'wb') as handle:
                    pickle.dump(results_dict, handle) 



if __name__ == '__main__':

    
    # Scaling experiment [todo]
    if False:
        scaling_experiment('Strogatz')
        scaling_experiment('Nguyen')
        scaling_experiment('Univ')
        scaling_experiment('Feynman')

    # Timing experiment [running]
    if True:
        timing_experiment('Strogatz')
        timing_experiment('Nguyen')
        timing_experiment('Univ')
        timing_experiment('Feynman')

    # Recovery experiment [done]
    if False:
        overwrite = True
        rand_state = 0
        problems = [n for n in os.listdir('datasets') if 'ipynb' not in n]
        problems = ['Nguyen', 'Strogatz', 'Feynman', 'Univ']

        regs = {
            #'linreg' : (regressors.LinReg(), True),
            #'polyreg2' : (regressors.PolyReg(degree= 2), True),
            #'polyreg3' : (regressors.PolyReg(degree= 3), True),
            #'MLP' : (regressors.MLP(random_state = rand_state), False),
            #'operon' : (regressors.Operon(random_state = rand_state), True),
            #'gplearn' : (regressors.GPlearn(random_state = rand_state), True),
            #'dsr' : (regressors.DSR(), True),
            'DAGSearch' : (dag_search.DAGRegressor(processes = 16, random_state = rand_state), True), 
            'DAGSearchPoly' : (dag_search.DAGRegressorPoly(processes = 16, random_state = rand_state), True), 
        }
        for ds_name in problems:
            for regressor_name in regs:
                if overwrite or (not os.path.exists(f'results/{ds_name}/{regressor_name}_results.p')):
                    regressor, is_symb = regs[regressor_name]
                    recovery_experiment(ds_name = ds_name, regressor = regressor, regressor_name = regressor_name, is_symb = is_symb)

    # ESR recovery experiment [done]
    if False:
        overwrite = False
        rand_state = 0
        ds_name = 'Univ'
        regs = {
            'esr' : (regressors.ESR(path_to_eqs = 'regressors/core_maths', max_complexity = 9, verbose = 2, random_state = rand_state), True) 
        }
        for regressor_name in regs:
            if overwrite or (not os.path.exists(f'results/{ds_name}/{regressor_name}_results.p')):
                regressor, is_symb = regs[regressor_name]
                recovery_experiment(ds_name = ds_name, regressor = regressor, regressor_name = regressor_name, is_symb = is_symb)

    # OLD - not used in the paper

    # Covariance experiment
    if False:
        covariance_experiment('Strogatz')
        covariance_experiment('Nguyen')
        covariance_experiment('Univ')
        covariance_experiment('Feynman')

    # proximity experiment
    if False:
        proximity_experiment(0)

    # Local Minima experiment
    if False:
        local_minima_experiment('Feynman')
        local_minima_experiment('Strogatz')
        local_minima_experiment('Nguyen')
        local_minima_experiment('Univ')

    # Fitness-distance correlation experiment
    if False:
        fdc_experiment('Feynman')
        fdc_experiment('Strogatz')
        fdc_experiment('Nguyen')
        fdc_experiment('Univ')
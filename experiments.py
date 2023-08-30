import numpy as np
import pickle
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.metrics import r2_score

from DAG_search import utils
from DAG_search import dag_search


def recovery_experiment(ds_name : str, mode : str = 'exhaustive', k : int = 1, topk : int = 5, processes : int = 1):
    '''
    Simple Experiment to estimate the Recovery rate of DAG-search.

    @Params:
        ds_name... Name of dataset
    '''
    
    load_path = f'datasets/{ds_name}/tasks.p'
    save_path = f'results/{ds_name}/{mode}_results.p'

    results = {}
    with open(load_path, 'rb') as handle:
        task_dict = pickle.load(handle)
    problems = list(task_dict.keys())

    for problem in problems:
        print('####################')
        print(problem)
        print('####################')

        X, y, exprs_true = task_dict[problem]['X'], task_dict[problem]['y'], task_dict[problem]['expr']
        total_rec = []
        best_graphs = []
        best_consts = []
        for idx in range(y.shape[1]):
            expr_true = exprs_true[idx]
            y_part = y[:, idx].reshape(-1, 1)

            m = X.shape[1]
            n = 1
            loss_fkt = dag_search.MSE_loss_fkt(y_part)

            if mode == 'exhaustive':
                # exhaustive search
                params = {
                    'X' : X,
                    'n_outps' : n,
                    'loss_fkt' : loss_fkt,
                    'k' : k,
                    'n_calc_nodes' : 4,
                    'n_processes' : processes,
                    'topk' : topk,
                    'opt_mode' : 'grid_zoom',
                    'verbose' : 2,
                    'max_orders' : int(5e5), 
                    'stop_thresh' : 1e-6
                }
                res = dag_search.exhaustive_search(**params)
            else:
                # sample search
                params = {
                    'X' : X,
                    'n_outps' : n,
                    'loss_fkt' : loss_fkt,
                    'k' : k,
                    'n_calc_nodes' : 5,
                    'n_processes' : processes,
                    'topk' : topk,
                    'opt_mode' : 'grid_zoom',
                    'verbose' : 2,
                    'n_samples' : 10000,
                    'stop_thresh' : 1e-6
                    
                }
                res = dag_search.sample_search(**params)


            recs = []
            for graph, consts in zip(res['graphs'], res['consts']):
                expr_est = graph.evaluate_symbolic(c = consts)[0]
                rec = utils.symb_eq(expr_est, expr_true) 
                recs.append(rec)
            total_rec.append(np.any(recs))
            best_graphs.append(res['graphs'][0])
            best_consts.append(res['consts'][0])
        
        est_exprs = [graph.evaluate_symbolic(c = consts) for graph, consts in zip(best_graphs, best_consts)]

        results[problem] = {
            'recovery' : total_rec,
            'exprs' : est_exprs,
            'graphs' : best_graphs,
            'consts' : best_consts
        }

        with open(save_path, 'wb') as handle:
            pickle.dump(results, handle)


def measure_experiment(random_state = None):
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

if __name__ == '__main__':

    np.random.seed(0)
    processes = 10

    if True:
        recovery_experiment('Nguyen', mode = 'sampling', k = 1, topk = 5, processes=processes)
        recovery_experiment('Strogatz', mode = 'sampling', k = 1, topk = 5, processes=processes)
        recovery_experiment('Feynman', mode = 'sampling', k = 1, topk = 5, processes=processes)

    if True:
        recovery_experiment('Nguyen', mode = 'exhaustive', k = 1, topk = 5, processes=processes)
        recovery_experiment('Strogatz', mode = 'exhaustive', k = 1, topk = 5, processes=processes)
        recovery_experiment('Feynman', mode = 'exhaustive', k = 1, topk = 5, processes=processes)

    if True:
        measure_experiment(0)
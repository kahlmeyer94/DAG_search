import numpy as np
import pickle
import dag_search
import utils


def recovery_experiment(ds_name : str, mode : str = 'exhaustive', k : int = 1, topk : int = 5, processes : int = 1):
    load_path = f'datasets/{ds_name}/tasks.p'
    save_path = f'datasets/{ds_name}/{mode}_results.p'

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
                    'n_calc_nodes' : 3,
                    'n_processes' : processes,
                    'topk' : topk,
                    'opt_mode' : 'grid_zoom',
                    'verbose' : 2,
                    'max_orders' : 50000, 
                    'stop_thresh' : 1e-4
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
                    'n_samples' : 100000,
                    'stop_thresh' : 1e-4
                    
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
            'exprs' : est_exprs
        }

        with open(save_path, 'wb') as handle:
            pickle.dump(results, handle)

if __name__ == '__main__':
    processes = 5

    recovery_experiment('Nguyen', mode = 'sampling', k = 1, topk = 5, processes=processes)
    recovery_experiment('Strogatz', mode = 'sampling', k = 1, topk = 5, processes=processes)
    recovery_experiment('Feynman', mode = 'sampling', k = 1, topk = 5, processes=processes)

    recovery_experiment('Nguyen', mode = 'exhaustive', k = 1, topk = 5, processes=processes)
    recovery_experiment('Strogatz', mode = 'exhaustive', k = 1, topk = 5, processes=processes)
    recovery_experiment('Feynman', mode = 'exhaustive', k = 1, topk = 5, processes=processes)
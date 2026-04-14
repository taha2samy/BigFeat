"""
base.py
-------
The main orchestrator class for BigFeat.
"""

import numpy as np
import ray

from sklearn.preprocessing import MinMaxScaler
import bigfeat.local_utils as local_utils
from .config import initialize_ray
from .distributed_tasks import remote_generate_batch, remote_get_importance

# Importing from our refactored modules
from .generator import feat_with_depth, feat_with_depth_gen
from .importance import get_feature_importances
from .tree_utils import get_paths, get_split_feats
from .selection import check_correlations, fit_fanova
from .evaluation import select_estimator as eval_select_estimator


class BigFeat:
    """Base BigFeat Class for both classification and regression tasks"""

    def __init__(self, task_type='classification', options=None):
        """
        Initialize the BigFeat object.
        
        Parameters:
        -----------
        task_type : str, default='classification'
            The type of machine learning task. Either 'classification' or 'regression'.
        options : dict, default=None
            Configuration for Ray initialization (address, cpus, etc.)
        """
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

        self.task_type = task_type
        self.options = options or {} 
        self.n_jobs = -1
        
        # Define operators
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]

    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg', 
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True):
        """
        Fit the BigFeat model using distributed Ray workers.
        """
        initialize_ray(self.options)
        
        self.selection = selection
        self.imp_operators = np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        
        # self.gen_steps = []
        self.n_feats = X.shape[1]
        self.n_rows = X.shape[0]
        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        
        # self.comb_mat = np.ones((self.n_feats, self.n_feats))
        self.split_vec = np.ones(self.n_feats)
        self.rng = np.random.RandomState(seed=random_state)
        
        iters_comb = np.zeros((self.n_rows, self.n_feats * iterations))
        depths_comb = np.zeros(self.n_feats * iterations)
        ids_comb = [None] * (self.n_feats * iterations)
        ops_comb = [None] * (self.n_feats * iterations)
        
        self.depth_range = np.arange(3) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()
        
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_ref = ray.put(X_scaled)
        y_ref = ray.put(y)
        
        if feat_imps:
            future = remote_get_importance.remote(
                X_ref, y_ref, estimator, self.task_type, random_state, self.n_jobs
            )
            importance_sum, splits_sum = ray.get(future)
            
            self.ig_vector = importance_sum / (importance_sum.sum() + 1e-9)
            self.split_vec += splits_sum
            self.split_vec /= self.split_vec.sum()
            
            # self.split_vec = StandardScaler().fit_transform(self.split_vec.reshape(1, -1), {'var_':5})
            
            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector, self.split_vec)
                self.ig_vector /= self.ig_vector.sum()
            elif split_feats == "splits":
                self.ig_vector = self.split_vec
                
        num_to_gen = self.n_feats * gen_size
        
        user_batch_size = self.options.get('batch_size', 'auto')
        if user_batch_size == 'auto':
            num_cpus = int(ray.cluster_resources().get("CPU", 1))
            batch_size = max(1, num_to_gen // (num_cpus * 2)) 
        else:
            batch_size = max(1, int(user_batch_size))

        for iteration in range(iterations):
            gen_futures = []
            
            for i in range(0, num_to_gen, batch_size):
                current_batch_size = min(batch_size, num_to_gen - i)
                batch_depths = [
                    self.rng.choice(self.depth_range, p=self.depth_weights) 
                    for _ in range(current_batch_size)
                ]
                
                task = remote_generate_batch.remote(
                    X_ref, batch_depths, random_state + i + iteration,
                    self.ig_vector, self.operators, self.operator_weights,
                    self.binary_operators, self.unary_operators
                )
                gen_futures.append(task)
            
            all_batches = ray.get(gen_futures)
            flattened_results = [item for batch in all_batches for item in batch]
            
            gen_feats_iter = np.column_stack([res[0] for res in flattened_results])
            iter_ops = [res[1] for res in flattened_results]
            iter_ids = [res[2] for res in flattened_results]
            iter_depths = np.array([res[3] for res in flattened_results])
            
            imps_iter, _ = get_feature_importances(
                gen_feats_iter, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs
            )
            feat_args = np.argsort(imps_iter)[-self.n_feats:]
            
            start_idx = iteration * self.n_feats
            end_idx = (iteration + 1) * self.n_feats
            
            iters_comb[:, start_idx:end_idx] = gen_feats_iter[:, feat_args]
            depths_comb[start_idx:end_idx] = iter_depths[feat_args]
            
            for k, idx in enumerate(feat_args):
                ids_comb[start_idx + k] = iter_ids[idx]
                ops_comb[start_idx + k] = iter_ops[idx]
                
            selected_ops = [iter_ops[idx] for idx in feat_args]
            for i, op in enumerate(self.operators):
                for feat in selected_ops:
                    for feat_op in feat:
                        if op == feat_op[0]:
                            self.imp_operators[i] += 1
            self.operator_weights = self.imp_operators / self.imp_operators.sum()
            
        if selection == 'stability' and iterations > 1 and combine_res:
            imps_final, _ = get_feature_importances(
                iters_comb, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs
            )
            feat_args = np.argsort(imps_final)[-self.n_feats:]
            gen_feats = iters_comb[:, feat_args]
            self.tracking_ids = [ids_comb[i] for i in feat_args]
            self.tracking_ops = [ops_comb[i] for i in feat_args]
            self.feat_depths = depths_comb[feat_args]
        else:
            gen_feats = iters_comb[:, -self.n_feats:]
            self.tracking_ids = ids_comb[-self.n_feats:]
            self.tracking_ops = ops_comb[-self.n_feats:]
            self.feat_depths = depths_comb[-self.n_feats:]

        if selection == 'stability' and check_corr:
            gen_feats, to_drop_cor = check_correlations(gen_feats)
            self.tracking_ids = [
                item for i, item in enumerate(self.tracking_ids) 
                if i not in to_drop_cor
            ]
            self.tracking_ops = [
                item for i, item in enumerate(self.tracking_ops) 
                if i not in to_drop_cor
            ]
            self.feat_depths = np.delete(self.feat_depths, to_drop_cor)
            
        gen_feats = np.hstack((gen_feats, X_scaled))

        if selection == 'fAnova':
            gen_feats, self.fAnova_best = fit_fanova(
                gen_feats, y, self.task_type, self.n_feats
            )

        return gen_feats

    def transform(self, X):
        """
        Produce features from the fitted BigFeat object.
        """
        X_scaled = self.scaler.transform(X)
        n_rows = X_scaled.shape[0]
        gen_feats = np.zeros((n_rows, len(self.tracking_ids)))

        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            # Copy lists to prevent modifying the fitted state during pop()
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()

            gen_feats[:, i] = feat_with_depth_gen(
                X_scaled, dpth, op_ls, id_ls, 
                self.binary_operators, self.unary_operators
            )

        gen_feats = np.hstack((gen_feats, X_scaled))

        if self.selection == 'fAnova':
            gen_feats = self.fAnova_best.transform(gen_feats)

        return gen_feats

    def select_estimator(self, X, y, estimators_names=None):
        """
        Select the best estimator based on cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        estimators_names : list or None
            List of estimator names to try.
        
        Returns:
        --------
        model : estimator
            Fitted best estimator.
        """
        return eval_select_estimator(
            X, y, 
            task_type=self.task_type, 
            n_jobs=self.n_jobs, 
            estimators_names=estimators_names
        )
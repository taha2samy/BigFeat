"""
base.py
-------
The main orchestrator class for BigFeat.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import bigfeat.local_utils as local_utils

# Importing from our refactored modules
from .generator import feat_with_depth, feat_with_depth_gen
from .importance import get_feature_importances
from .tree_utils import get_paths, get_split_feats
from .selection import check_correlations, fit_fanova
from .evaluation import select_estimator as eval_select_estimator


class BigFeat:
    """Base BigFeat Class for both classification and regression tasks"""

    def __init__(self, task_type='classification'):
        """
        Initialize the BigFeat object
        
        Parameters:
        -----------
        task_type : str, default='classification'
            The type of machine learning task. Either 'classification' or 'regression'.
        """
        self.n_jobs = -1
        self.operators = [np.multiply, np.add, np.subtract, np.abs, np.square]
        self.binary_operators = [np.multiply, np.add, np.subtract]
        self.unary_operators = [np.abs, np.square, local_utils.original_feat]
        self.task_type = task_type
        
        # Validate task_type input
        if task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")

    def fit(self, X, y, gen_size=5, random_state=0, iterations=5, estimator='avg', 
            feat_imps=True, split_feats=None, check_corr=True, selection='stability', combine_res=True):
        """ Generated Features using test set """
        self.selection = selection
        self.imp_operators = np.ones(len(self.operators))
        self.operator_weights = self.imp_operators / self.imp_operators.sum()
        # self.gen_steps = []
        self.n_feats = X.shape[1]
        self.n_rows = X.shape[0]
        self.ig_vector = np.ones(self.n_feats) / self.n_feats
        #self.comb_mat = np.ones((self.n_feats, self.n_feats))
        self.split_vec = np.ones(self.n_feats)
        # Set RNG seed if provided for numpy
        self.rng = np.random.RandomState(seed=random_state)
        gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
        iters_comb = np.zeros((self.n_rows, self.n_feats * iterations))
        depths_comb = np.zeros(self.n_feats * iterations)
        ids_comb = np.zeros(self.n_feats * iterations, dtype=object)
        ops_comb = np.zeros(self.n_feats * iterations, dtype=object)
        self.feat_depths = np.zeros(gen_feats.shape[1])
        self.depth_range = np.arange(3) + 1
        self.depth_weights = 1 / (2 ** self.depth_range)
        self.depth_weights /= self.depth_weights.sum()
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        
        if feat_imps:
            self.ig_vector, estimators = get_feature_importances(
                X, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs
            )
            self.ig_vector /= self.ig_vector.sum()
            for tree in estimators:
                paths = get_paths(tree, np.arange(X.shape[1]))
                get_split_feats(paths, self.split_vec)
            self.split_vec /= self.split_vec.sum()
            # self.split_vec = StandardScaler().fit_transform(self.split_vec.reshape(1, -1), {'var_':5})
            if split_feats == "comb":
                self.ig_vector = np.multiply(self.ig_vector, self.split_vec)
                self.ig_vector /= self.ig_vector.sum()
            elif split_feats == "splits":
                self.ig_vector = self.split_vec
                
        for iteration in range(iterations):
            self.tracking_ops = []
            self.tracking_ids = []
            gen_feats = np.zeros((self.n_rows, self.n_feats * gen_size))
            self.feat_depths = np.zeros(gen_feats.shape[1])
            for i in range(gen_feats.shape[1]):
                dpth = self.rng.choice(self.depth_range, p=self.depth_weights)
                ops = []
                ids = []
                gen_feats[:, i] = feat_with_depth(
                    X, dpth, ops, ids, self.rng, self.ig_vector, 
                    self.operators, self.operator_weights, 
                    self.binary_operators, self.unary_operators
                )  # ops and ids are updated
                self.feat_depths[i] = dpth
                self.tracking_ops.append(ops)
                self.tracking_ids.append(ids)
            self.tracking_ids = list(self.tracking_ids)
            self.tracking_ops = list(self.tracking_ops)
            
            imps, estimators = get_feature_importances(
                gen_feats, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs
            )
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = gen_feats[:, feat_args]
            self.tracking_ids = self.tracking_ids[feat_args]
            self.tracking_ops = self.tracking_ops[feat_args]
            self.feat_depths = self.feat_depths[feat_args]
            depths_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.feat_depths
            ids_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ids
            ops_comb[iteration * self.n_feats:(iteration + 1) * self.n_feats] = self.tracking_ops
            iters_comb[:, iteration * self.n_feats:(iteration + 1) * self.n_feats] = gen_feats
            for i, op in enumerate(self.operators):
                for feat in self.tracking_ops:
                    for feat_op in feat:
                        if op == feat_op[0]:
                            self.imp_operators[i] += 1
            self.operator_weights = self.imp_operators / self.imp_operators.sum()
            
        if selection == 'stability' and iterations > 1 and combine_res:
            imps, estimators = get_feature_importances(
                iters_comb, y, estimator, random_state, self.task_type, n_jobs=self.n_jobs
            )
            total_feats = np.argsort(imps)
            feat_args = total_feats[-self.n_feats:]
            gen_feats = iters_comb[:, feat_args]
            self.tracking_ids = [ids_comb[i] for i in feat_args]
            self.tracking_ops = [ops_comb[i] for i in feat_args]
            self.feat_depths = depths_comb[feat_args]

        if selection == 'stability' and check_corr:
            gen_feats, to_drop_cor = check_correlations(gen_feats)
            self.tracking_ids = [item for i, item in enumerate(self.tracking_ids) if i not in to_drop_cor]
            self.tracking_ops = [item for i, item in enumerate(self.tracking_ops) if i not in to_drop_cor]
            self.feat_depths = np.delete(self.feat_depths, to_drop_cor)
            
        gen_feats = np.hstack((gen_feats, X))

        if selection == 'fAnova':
            gen_feats, self.fAnova_best = fit_fanova(gen_feats, y, self.task_type, self.n_feats)

        return gen_feats

    def transform(self, X):
        """ Produce features from the fitted BigFeat object """
        X = self.scaler.transform(X)
        self.n_rows = X.shape[0]
        gen_feats = np.zeros((self.n_rows, len(self.tracking_ids)))
        for i in range(gen_feats.shape[1]):
            dpth = self.feat_depths[i]
            op_ls = self.tracking_ops[i].copy()
            id_ls = self.tracking_ids[i].copy()
            gen_feats[:, i] = feat_with_depth_gen(
                X, dpth, op_ls, id_ls, self.binary_operators, self.unary_operators
            )
        gen_feats = np.hstack((gen_feats, X))
        if self.selection == 'fAnova':
            gen_feats = self.fAnova_best.transform(gen_feats)
        return gen_feats

    def select_estimator(self, X, y, estimators_names=None):
        """
        Select the best estimator based on cross-validation
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        estimators_names : list or None
            List of estimator names to try. If None, uses appropriate defaults.
        
        Returns:
        --------
        model : estimator
            Fitted best estimator
        """
        return eval_select_estimator(
            X, y, task_type=self.task_type, n_jobs=self.n_jobs, estimators_names=estimators_names
        )
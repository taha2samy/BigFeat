"""
distributed_tasks.py
--------------------
Wraps core functions into Ray remote tasks for distributed execution.
"""

import ray
import numpy as np
from .generator import feat_with_depth
from .importance import get_feature_importances

@ray.remote
def remote_generate_batch(X_ref, depths, rng_seed, ig_vector, operators, 
                          op_weights, binary_ops, unary_ops):
    """
    Generates a batch of features on a remote worker.
    Using X_ref (Object Store reference) to save memory.
    """
    import numpy as np # Vital for remote workers
    rng = np.random.RandomState(rng_seed)
    batch_results = []
    
    for dpth in depths:
        ops = []
        ids = []
        # Calling our standard generator function
        feat_column = feat_with_depth(
            X_ref, dpth, ops, ids, rng, ig_vector, 
            operators, op_weights, binary_ops, unary_ops
        )
        batch_results.append((feat_column, ops, ids, dpth))
        
    return batch_results

@ray.remote
def remote_get_importance(X_sample, y_sample, estimator, task_type, random_seed, n_jobs):
    """
    Calculates importance on a remote worker.
    """
    # Calling our standard importance function
    # Note: We return importance_sum and split_vec (The vector approach!)
    from .importance import get_feature_importances
    from .tree_utils import get_paths, get_split_feats
    
    imps, estimators = get_feature_importances(
        X_sample, y_sample, estimator, random_seed, task_type, n_jobs=n_jobs
    )
    
    # Calculate splits locally on the worker to avoid sending heavy tree objects
    split_vec = np.zeros(X_sample.shape[1])
    for tree in estimators:
        paths = get_paths(tree, np.arange(X_sample.shape[1]))
        get_split_feats(paths, split_vec)
        
    return imps, split_vec
"""
evaluation.py
-------------
Responsible for model evaluation and selecting the best estimator.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, r2_score, make_scorer


def select_estimator(X, y, task_type='classification', n_jobs=-1, estimators_names=None):
    """
    Select the best estimator based on cross-validation
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    task_type : str, default='classification'
        The type of machine learning task. Either 'classification' or 'regression'.
    n_jobs : int, default=-1
        The number of jobs to run in parallel.
    estimators_names : list or None
        List of estimator names to try. If None, uses appropriate defaults.
    
    Returns:
    --------
    model : estimator
        Fitted best estimator
    """
    # Use appropriate default estimators based on task type
    if estimators_names is None:
        if task_type == 'classification':
            estimators_names = ['dt', 'lr']
        else:  # regression
            estimators_names = ['dt_reg', 'lr_reg']
    
    # Define available estimators based on task type
    estimators_dic = {
        # Classification estimators
        'dt': DecisionTreeClassifier(),
        'lr': LogisticRegression(),
        'rf': RandomForestClassifier(n_jobs=n_jobs),
        'lgb': LGBMClassifier(),
        
        # Regression estimators
        'dt_reg': DecisionTreeRegressor(),
        'lr_reg': LinearRegression(),
        'rf_reg': RandomForestRegressor(n_jobs=n_jobs),
        'lgb_reg': LGBMRegressor()
    }
    
    models_score = {}

    for estimator in estimators_names:
        model = estimators_dic[estimator]
        
        # Use appropriate scoring metric based on task type
        if task_type == 'classification':
            scorer = make_scorer(f1_score)
        else:  # regression
            scorer = make_scorer(r2_score)
            
        models_score[estimator] = cross_val_score(model, X, y, cv=3, scoring=scorer).mean()
        
    best_estimator = max(models_score, key=models_score.get)
    best_model = estimators_dic[best_estimator]
    best_model.fit(X, y)
    
    return best_model
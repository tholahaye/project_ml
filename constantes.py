#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#import streamlit as st

STRUCTURE = {
    'classification': {
        'Decision_Tree': {
            'model': DecisionTreeClassifier(),
            'hyperparameters': {
                'criterion': {
                    'type': str,
                    'description': "The optimized criterion for node division",
                    'values': ['gini', 'entropy']
                },
                # splitter: {},
                'max_depth': {
                    'type': int,
                    'description': "(integer) The maximum depth of the tree",
                    'default': None,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                },
                'min_sample_split': {
                    'type': int,
                    'description': "(integer) The minimum number of data objects required to split a node",
                    'default': 2,
                    'min value': 2,
                    'max_value': np.inf,
                    'values': None
                },
                'min_samples_leaf': {
                    'type': int,
                    'description': "(integer) The minimum number of data objects required on leaves",
                    'default': 1,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                },
                'max_leaf_nodes': {
                    'type': int,
                    'description': "(integer) The maximum number of leaves",
                    'default': None,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                }
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {
                'n_estimators': {
                    'type': int,
                    'description': "The number of trees",
                    'default': 100,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                },
                'max_features': {
                    'type': int,
                    'description': "The number of features to sample on each tree",
                    'default': None, #TODO if none or empty char, then set to "sqrt"
                    'min value': 1,
                    'max_value': np.inf, #TODO len(df.columns),
                    'values': None
                },
                'criterion': {
                    'type': str,
                    'description': "The optimized criterion for node division",
                    'values': ['gini', 'entropy']
                },
                # splitter: {},
                'max_depth': {  #The maximum depth of the tree
                    'type': int,
                    'description': "(integer) The maximum tree depth",
                    'default': None,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                },
                'min_sample_split': {
                    'type': int,
                    'description': "(integer) The minimum number of data objects required to split a node",
                    'default': 2,
                    'min value': 2,
                    'max_value': np.inf,
                    'values': None
                },
                'min_samples_leaf': {
                    'type': int,
                    'description': "(integer) The minimum number of data objects required on leaves",
                    'default': 1,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                },
                'max_leaf_nodes': {
                    'type': int,
                    'description': "(integer) The maximum number of leaves",
                    'default': None,
                    'min value': 1,
                    'max_value': np.inf,
                    'values': None
                }
            }
        }
    #     ,
    #     'Regression_Logistique': {
    #         'model': LogisticRegression(),
    #         'hyperparameters': {
    #             'penalty': {
    #                 'type': str,
    #                 'values': ['l1', 'l2', 'elasticnet', 'None'],
    #             },
    #             'C': {
    #                 'type': int,
    #                 'values': [1, 5, 10],
    #             },
    #             'solver': {
    #                 'type': str,
    #                 'values': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    #             }
    #         }
    #     },
    #     'KNeighbours_Classifier': {
    #         'model': KNeighborsClassifier(),
    #         'hyperparameters': {}
    #     },
    #     'SVC': {
    #         'model': SVC(),
    #         'hyperparameters': {}
    #     },
    },
    'regression': {
        'Linear_Regression': {
            'model': LinearRegression(),
            'hyperparameters': {}
        },
        'Ridge_Regression': {
            'model': Ridge(),
            'hyperparameters': {
                'alpha': {
                    'type': int,
                    'values': [0, 1, 100],
                },
            }
        },
    }
}
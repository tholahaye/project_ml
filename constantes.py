#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#import streamlit as st

structure = {
    'Classification': {
        'Decision_Tree': {
            'model': DecisionTreeClassifier(),
            'hyperparameters': {
                'criterion': {
                    'type': 'str',
                    'description': "The optimized criterion for node division",
                    'values': ['gini', 'entropy']
                },
                # splitter: {},
                'max_depth': {
                    'type': 'int',
                    'description': "(positive integer) The maximum depth of the tree",
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf'),
                    # 'values': None
                },
                'min_samples_split': {
                    'type': 'int',
                    'description': "(positive integer > 1) The minimum number of data objects required to split a node",
                    'default': 2,
                    'min_value': 2,
                    'max_value': float('inf'),
                    # 'values': None
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'description': "(positive integer) The minimum number of data objects required on leaves",
                    'default': 1,
                    'min_value': 1,
                    'max_value': float('inf'),
                    # 'values': None
                },
                'max_leaf_nodes': {
                    'type': 'int',
                    'description': "(positive integer) The maximum number of leaves",
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf'),
                    # 'values': None
                }
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {
                'n_estimators': {
                    'type': 'int',
                    'description': "(positive integer) The number of trees",
                    'default': 100,
                    'min_value': 1,
                    'max_value': float('inf'),
                    # 'values': None
                },
                'max_features': {
                    'type': 'int',
                    'description': "(positive integer) The number of features to sample on each tree",
                    'default': None, #TODO if none or empty char, then set to "sqrt"
                    'min value': 1,
                    'max_value': float('inf'), #TODO len(df.columns) #,
                    # 'values': None
                },
                'criterion': {
                    'type': 'str',
                    'description': "The optimized criterion for node division",
                    'values': ['gini', 'entropy']
                },
                # splitter: {},
                'max_depth': {  #The maximum depth of the tree
                    'type': 'int',
                    'description': "(positive integer) The maximum tree depth",
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf'),
                    #'values': None
                },
                'min_samples_split': {
                    'type': 'int',
                    'description': "(positive integer > 1) The minimum number of data objects required to split a node",
                    'default': 2,
                    'min_value': 2,
                    'max_value': float('inf'),
                    #'values': None
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'description': "(positive integer) The minimum number of data objects required on leaves",
                    'default': 1,
                    'min_value': 1,
                    'max_value': float('inf'),
                    #'values': None
                },
                'max_leaf_nodes': {
                    'type': 'int',
                    'description': "(positive integer) The maximum number of leaves",
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf'),
                    #'values': None
                }
            }
        }
    #     ,
    #     'Regression_Logistique': {
    #         'model': LogisticRegression(),
    #         'hyperparameters': {
    #             'penalty': {
    #                 'type': 'str',
    #                 'values': ['l1', 'l2', 'elasticnet', 'None'],
    #             },
    #             'C': {
    #                 'type': 'int',
    #                 'values': [1, 5, 10],
    #             },
    #             'solver': {
    #                 'type': 'str',
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
    'Regression': {
        'Linear_Regression': {
            'model': LinearRegression(),
            'hyperparameters': {}
        },
        'Ridge_Regression': {
            'model': Ridge(),
            'hyperparameters': {
                'alpha': {
                    'type': 'float',
                    'description': "(non negative float) Regularization constant, use linear regression instead of 0",
                    'default': 1,
                    'min_value': 0,
                    'max_value': float('inf'),
                    # 'values': [0, 1, 100]
                },
            }
        },
    }
}
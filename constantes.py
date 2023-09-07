#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge


TARGET = 'target'

CV_SCORES = {
    'Classification': ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
    'Regression': ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'max_error']
}

CV_MAX_RES = 10

STRUCTURE = {
    'Classification': {
        'Decision_Tree': {
            'model': DecisionTreeClassifier(),
            'hyperparameters': {
                'criterion': {
                    'type': 'str',
                    'description': "The optimized criterion for node division",
                    'optional': False,
                    'values': ['gini', 'entropy']
                },
                # splitter: {},
                'max_depth': {
                    'type': 'int',
                    'description': "(positive integer) The maximum depth of the tree",
                    'optional': True,
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf')
                },
                'min_samples_split': {
                    'type': 'int',
                    'description': "(positive integer > 1) The minimum number of data objects required to split a node",
                    'optional': False,
                    'default': 2,
                    'min_value': 2,
                    'max_value': float('inf')
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'description': "(positive integer) The minimum number of data objects required on leaves",
                    'optional': False,
                    'default': 1,
                    'min_value': 1,
                    'max_value': float('inf')
                },
                'max_leaf_nodes': {
                    'type': 'int',
                    'description': "(positive integer) The maximum number of leaves",
                    'optional': True,
                    'default': None,
                    'min_value': 2,
                    'max_value': float('inf')
                }
            }
        },
        'Random_Forest': {
            'model': RandomForestClassifier(),
            'hyperparameters': {
                'n_estimators': {
                    'type': 'int',
                    'description': "(positive integer) The number of trees",
                    'optional': False,
                    'default': 100,
                    'min_value': 1,
                    'max_value': float('inf')
                },
                'max_features': {
                    'type': 'int',
                    'description': "(positive integer) The number of features to sample on each tree. If not specified, then at most n^(1/2) features are sampled.",
                    'optional': True,
                    'default': "sqrt",  # TODO if not selected or empty number, then set to "sqrt"
                    'min_value': 1,
                    'max_value': float('inf')  # TODO len(df.columns)
                },
                'criterion': {
                    'type': 'str',
                    'description': "The optimized criterion for node division",
                    'optional': False,
                    'values': ['gini', 'entropy']
                },
                # splitter: {},
                'max_depth': {  # The maximum depth of the tree
                    'type': 'int',
                    'description': "(positive integer) The maximum tree depth",
                    'optional': True,
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf')
                },
                'min_samples_split': {
                    'type': 'int',
                    'description': "(positive integer > 1) The minimum number of data objects required to split a node",
                    'optional': False,
                    'default': 2,
                    'min_value': 2,
                    'max_value': float('inf')
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'description': "(positive integer) The minimum number of data objects required on leaves",
                    'optional': False,
                    'default': 1,
                    'min_value': 1,
                    'max_value': float('inf')
                },
                'max_leaf_nodes': {
                    'type': 'int',
                    'description': "(positive integer) The maximum number of leaves",
                    'optional': True,
                    'default': None,
                    'min_value': 1,
                    'max_value': float('inf')
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
                    'optional': False,
                    'default': 1,
                    'min_value': 0,
                    'max_value': float('inf'),
                },
            }
        },
    }
}
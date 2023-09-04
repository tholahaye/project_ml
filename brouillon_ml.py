import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

import constantes
TARGET = 'target'

df = pd.read_csv("data/diabete.csv")
print("dataset:", df.head())
model_type = "classification"


X = df.drop(columns=[TARGET])
print("features:", X.head())
y = df[TARGET]
print("target:", y.head())

if model_type == "classification":
    classes = set(y.unique())
print("whole dataset classes:", classes)
# Todo for check imbalanced dataset


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
if model_type == "classification":
    classes_train = set(y_train.unique())
    if classes != classes_train:
        pass
        #TODO error message : "Training data do not have all classes represented, please change your random seed"
print("train dataset classes:", classes_train)
# Todo for check imbalanced dataset


if model_type == "classification":
    tab_eval = pd.DataFrame(columns=["model", "hyperparameters", "fold", "classe", "accuracy", "precision", "recall", "f1-score"])
if model_type == "regression":
    tab_eval = pd.DataFrame(columns=["model", "hyperparameters", "fold", "rmse", "mse", "mae"])

#print(tab_eval.head())

model_name = "Decision_Tree"
model = constantes.STRUCTURE[model_type][model_name]["model"]
#hyperParams = STRUCTURE[model_type][model_name]["hyperparameters"]
hyperParams = {"criterion": 'gini',
               "max_depth": None,
               "min_samples_split": 2,
               "min_samples_leaf": 1,
               "max_leaf_nodes": None}
model.set_params(**hyperParams)

'''
import itertools as it
values_lists = list(hyperParams.selValues())
combinations = list(it.product(*values_lists))
possibility = [dict(zip(parameters.keys(), combo)) for combo in combinations]
'''


fold_number = 1
try:
    model.fit(X_train, y_train)
except:
    print(f'Cannot fit {model_name} with {hyperParams}')





y_pred = model.predict(X_test)

if model_type == "classification":
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(report_dict)

    for cl in classes:
        row = {"model": model_name,
               "hyperparameters": hyperParams,
               "fold": fold_number,
               "classe": cl,
               #"accuracy": report_dict[cl]['f1-score'],
               "precision": report_dict[cl]['precision'],
               "recall": report_dict[cl]['recall'],
               "f1-score": report_dict[cl]['f1-score']}
        tab_eval = pd.concat([tab_eval, pd.DataFrame([row])], ignore_index=True)

    row = {"model": model_name,
           "hyperparameters": hyperParams,
           "fold": fold_number,
           "classe": "__all (macro avg)",
           "accuracy": report_dict["accuracy"],
           "precision": report_dict['macro avg']['precision'],
           "recall": report_dict['macro avg']['recall'],
           "f1-score": report_dict['macro avg']['f1-score']}
    tab_eval = pd.concat([tab_eval, pd.DataFrame([row])], ignore_index=True)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(tab_eval)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, max_error

from constantes import STRUCTURE, TARGET

df = pd.read_csv("data/diabete.csv")
model_type = "Classification"
#df = pd.read_csv("data/vin.csv")
#model_type = "Regression"

print("dataset:", df.head())

X = df.drop(columns=[TARGET])
print("features:", X.head())
y = df[TARGET]
print("target:", y.head())

if model_type == "Classification":
    classes = set(y.unique())
    print("whole dataset classes:", classes)
    # Todo for check imbalanced dataset


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
if model_type == "Classification":
    classes_train = set(y_train.unique())
    if classes != classes_train:
        pass
        #TODO error message : "Training data do not have all classes represented, please change your random seed"
        print("train dataset classes:", classes_train)
        # Todo for check imbalanced dataset


if model_type == "Classification":
    tab_eval = pd.DataFrame(columns=["model", "hyperparameters", "fold", "classe", "accuracy", "precision", "recall", "f1-score"])
if model_type == "Regression":
    tab_eval = pd.DataFrame(columns=["model", "hyperparameters", "fold", "rmse", "mae"])


model_name = "Decision_Tree"
# hyperParams = STRUCTURE[model_type][model_name]["hyperparameters"]
hyperParams = {"criterion": 'gini',
               "max_depth": None,
               "min_samples_split": 2,
               "min_samples_leaf": 1,
               "max_leaf_nodes": None}

#model_name = "Ridge_Regression"
# hyperParams = STRUCTURE[model_type][model_name]["hyperparameters"]
#hyperParams = {"alpha": 1}

model = STRUCTURE[model_type][model_name]["model"]
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

if model_type == "Classification":
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(report_dict)

    for cl in classes:
        row = {"model": model_name,
               "hyperparameters": hyperParams,
               #"fold": fold_number,
               "classe": cl,
               "precision": report_dict[cl]['precision'],
               "recall": report_dict[cl]['recall'],
               "f1-score": report_dict[cl]['f1-score']}
        tab_eval = pd.concat([tab_eval, pd.DataFrame([row])], ignore_index=True)

    row = {"model": model_name,
           "hyperparameters": hyperParams,
           #"fold": fold_number,
           "classe": "__all (macro avg)",
           "accuracy": report_dict["accuracy"],
           "precision": report_dict['macro avg']['precision'],
           "recall": report_dict['macro avg']['recall'],
           "f1-score": report_dict['macro avg']['f1-score']}
    tab_eval = pd.concat([tab_eval, pd.DataFrame([row])], ignore_index=True)

if model_type == "Regression":
    row = {"model": model_name,
           "hyperparameters": hyperParams,
           #"fold": fold_number,
           "rmse": mean_squared_error(y_test, y_pred),
           "mae": mean_absolute_error(y_test, y_pred),
           "maxe": max_error(y_test, y_pred)}
    tab_eval = pd.concat([tab_eval, pd.DataFrame([row])], ignore_index=True)

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(tab_eval)
import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, max_error

from constantes import STRUCTURE

#recupérer les classes, le model_type, model_name, hyperParams, X/y test & train



def learning(X, y, model):
    model.fit(X, y)
    return(model)

def predict(X, model):
    y_pred = model.predict(X)
    return(y_pred)


def create_tabEval_clf():
    tab_eval = pd.DataFrame(
        columns=["model", "hyperparameters", "fold", "classe", "accuracy", "precision", "recall", "f1-score"])
    return (tab_eval)

def evaluate_clf(y_true, y_pred, classesSet, modelName, hyperParams, tabEval):
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    for cl in classesSet:
        row = {"model": modelName,
               "hyperparameters": hyperParams,
               # "fold": fold_number,
               "classe": cl,
               "precision": report_dict[cl]['precision'],
               "recall": report_dict[cl]['recall'],
               "f1-score": report_dict[cl]['f1-score']}
        tabEval = pd.concat([tabEval, pd.DataFrame([row])], ignore_index=True)
    row = {"model": modelName,
            "hyperparameters": hyperParams,
            # "fold": fold_number,
            "classe": "__all (macro avg)",
            "accuracy": report_dict["accuracy"],
            "precision": report_dict['macro avg']['precision'],
            "recall": report_dict['macro avg']['recall'],
            "f1-score": report_dict['macro avg']['f1-score']}
    tabEval = pd.concat([tabEval, pd.DataFrame([row])], ignore_index=True)
    return tabEval


def create_tabEval_reg():
    tab_eval = pd.DataFrame(columns=["model", "hyperparameters", "fold", "rmse", "mae"])
    return (tab_eval)


def evaluate_reg(y_true, y_pred, modelName, hyperParams, tabEval):
    row = {"model": modelName,
           "hyperparameters": hyperParams,
           # "fold": fold_number,
           "rmse": mean_squared_error(y_true, y_pred),
           "mae": mean_absolute_error(y_true, y_pred),
           "maxe": max_error(y_true, y_pred)}
    tabEval = pd.concat([tabEval, pd.DataFrame([row])], ignore_index=True)
    return tabEval

def main(Structure = STRUCTURE, model_type, model_name, hyperParams, X_train, y_train, X_test, y_test):
    model = Structure[model_type][model_name]["model"]
    model.set_params(**hyperParams)
    model = learning(X=X_train, y=y_train, model=model)
    y_pred = predict(X=X_test, model=model)

    if model_type == "Classification":
        tab_eval = create_tabEval_clf()
        evaluate_clf(y_true=y_test, y_pred=y_pred, classesSet=classes,
                     modelName=model_name, hyperParams=hyperParams, tabEval=tab_eval)
    if model_type == "Regression":
        tab_eval = create_tabEval_reg()
        evaluate_reg(y_true=y_test, y_pred=y_pred,
                     modelName=model_name, hyperParams=hyperParams, tabEval=tab_eval)

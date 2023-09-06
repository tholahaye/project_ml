import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error,\
                            mean_absolute_error, max_error, confusion_matrix

from constantes import STRUCTURE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import streamlit as st

#recupérer les classes, le model_type, model_name, self.hyper_params, X/y test & train


class MachineLearning:
    def __init__(self, model_type, model_name, hyper_params, X_train, y_train, X_test, y_test, classes, cross_val):
        self.model_type = model_type
        self.model_name = model_name
        self.hyper_params = hyper_params
        # TODO? Changer X_train, X_test, y_train, y_test en un seul dictionnaire
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.classes = classes
        self.cross_val = cross_val

        # TODO: *********************CROSS-VAL ICI********************************************
        if self.cross_val:
            pass
        # TODO: ******************************************************************************
        else:
            self.model = STRUCTURE[self.model_type][self.model_name]["model"]
            self.model.set_params(**self.hyper_params)
            self.model = self.learning(X=self.X_train, y=self.y_train, model=self.model)
            self.y_pred = self.predict(X=self.X_test, model=self.model)

            if self.model_type == "Classification":
                self.tab_eval = self.create_tab_eval_clf()
                print(self.y_pred)
                print(self.y_test)
                self.evaluate_clf()
                self.cf_matrix = confusion_matrix(self.y_test, y_pred=self.y_pred)
            if self.model_type == "Regression":
                self.tab_eval = create_tab_eval_reg()
                self.evaluate_reg()


    def learning(self, X, y, model):
        model.fit(X, y)
        return model

    def predict(self, X, model):
        y_pred = model.predict(X)
        return y_pred

    def create_tab_eval_clf(self):
        tab_eval = pd.DataFrame(
            columns=["model", "hyperparameters", "fold", "classe",
                     "accuracy", "precision", "recall", "f1-score"])
        return tab_eval

    def evaluate_clf(self):
        report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)
        for cl in self.classes:
            row = {"model": self.model_name,
                   "hyperparameters": self.hyper_params,
                   # "fold": fold_number,
                   "classe": cl,
                   "precision": report_dict[cl]['precision'],
                   "recall": report_dict[cl]['recall'],
                   "f1-score": report_dict[cl]['f1-score']}
            self.tab_eval = pd.concat([self.tab_eval, pd.DataFrame([row])], ignore_index=True)
        row = {"model": self.model_name,
                "hyperparameters": self.hyper_params,
                # "fold": fold_number,
                "classe": "__all (macro avg)",
                "accuracy": report_dict["accuracy"],
                "precision": report_dict['macro avg']['precision'],
                "recall": report_dict['macro avg']['recall'],
                "f1-score": report_dict['macro avg']['f1-score']}
        self.tab_eval = pd.concat([self.tab_eval, pd.DataFrame([row])], ignore_index=True)

    def evaluate_reg(self):
        row = {"model": self.model_name,
               "hyperparameters": self.hyper_params,
               # "fold": fold_number,
               "rmse": mean_squared_error(self.y_test, self.y_pred),
               "mae": mean_absolute_error(self.y_test, self.y_pred),
               "maxe": max_error(self.y_test, self.y_pred)}
        self.tab_eval = pd.concat([self.tab_eval, pd.DataFrame([row])], ignore_index=True)
    
    def conf_matrix(self):

        # Créez un graphique de la matrice de confusion
        fig = plt.figure(figsize=(6, 6))
        sns.heatmap(self.cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
        plt.xlabel('Valeurs Prédites')
        plt.ylabel('Valeurs Réelles')
        plt.title('Matrice de Confusion')
        plt.show()
        return st.pyplot(fig)

def create_tab_eval_reg():
    tab_eval = pd.DataFrame(columns=["model", "hyperparameters", "fold", "rmse", "mae"])
    return tab_eval

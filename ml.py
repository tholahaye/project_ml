import pandas as pd
from sklearn.metrics import classification_report, mean_squared_error,\
                            mean_absolute_error, max_error, confusion_matrix
from sklearn.model_selection import GridSearchCV

from constantes import STRUCTURE, CV_SCORES
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


class NbHyperError(Exception):
    pass

class MachineLearning:
    def __init__(self, model_type, model_name, hyper_params, X_train, y_train, X_test, y_test, classes,
                 cross_val, cv_nfold, cv_score):
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
        self.cv_nfold = cv_nfold
        self.cv_score = cv_score


        self.model = STRUCTURE[self.model_type][self.model_name]["model"]

        # TODO: *********************CROSS-VAL ICI********************************************
        if self.cross_val:
            nb_hyper = set()
            for h in self.hyper_params.values():
                nb_hyper.add(len(h))
            if max(nb_hyper) <= 1:
                raise NbHyperError
            #TODO : erreur si une valeur unique par hyperparametre
            self.grid_search = GridSearchCV(self.model, self.hyper_params, cv=self.cv_nfold,
                                            scoring=CV_SCORES[self.model_type], refit=self.cv_score)
            self.grid_search.fit(self.X_train, self.y_train)
            self.cv_comb_params = self.grid_search.cv_results_['params']
            self.cv_ncomb_params = len(self.cv_comb_params)
            self.cv_best_params = self.grid_search.best_params_
            self.y_pred = self.grid_search.predict(self.X_test)
            self.cv_tab_eval = create_tab_eval_crossval(CV_SCORES[self.model_type])
            self.evaluate_crossval()
            if self.model_type == "Classification":
                self.tab_eval = create_tab_eval_clf()
                self.evaluate_clf()
                self.cf_matrix = confusion_matrix(self.y_test, y_pred=self.y_pred)
            if self.model_type == "Regression":
                self.tab_eval = create_tab_eval_reg()
                self.evaluate_reg()

        # TODO: ******************************************************************************
        else:
            self.model.set_params(**self.hyper_params)
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            if self.model_type == "Classification":
                self.tab_eval = create_tab_eval_clf()
                self.evaluate_clf()
                self.cf_matrix = confusion_matrix(self.y_test, y_pred=self.y_pred)
            if self.model_type == "Regression":
                self.tab_eval = create_tab_eval_reg()
                self.evaluate_reg()




    def evaluate_clf(self):
        if self.cross_val:
            params = self.cv_best_params
        else:
            params = self.hyper_params
        report_dict = classification_report(self.y_test, self.y_pred, output_dict=True)
        for cl in self.classes:
            row = {"hyperparameters": params,
                   "classe": cl,
                   "precision": report_dict[cl]['precision'],
                   "recall": report_dict[cl]['recall'],
                   "f1-score": report_dict[cl]['f1-score']}
            self.tab_eval = pd.concat([self.tab_eval, pd.DataFrame([row])], ignore_index=True)
        row = {"hyperparameters": params,
               "classe": "__all (macro avg)",
               "accuracy": report_dict["accuracy"],
               "precision": report_dict['macro avg']['precision'],
               "recall": report_dict['macro avg']['recall'],
               "f1-score": report_dict['macro avg']['f1-score']}
        self.tab_eval = pd.concat([self.tab_eval, pd.DataFrame([row])], ignore_index=True)

    def evaluate_reg(self):
        if self.cross_val:
            params = self.cv_best_params
        else:
            params = self.hyper_params
        row = {"hyperparameters": params,
               "rmse": mean_squared_error(self.y_test, self.y_pred),
               "mae": mean_absolute_error(self.y_test, self.y_pred),
               "maxerror": max_error(self.y_test, self.y_pred)}
        self.tab_eval = pd.concat([self.tab_eval, pd.DataFrame([row])], ignore_index=True)

    def evaluate_crossval(self):
        for i in range(len(self.cv_comb_params)):
            row = {"hyperparameters": self.cv_comb_params[i]}
            for score in CV_SCORES[self.model_type]:
                row[score] = self.grid_search.cv_results_["mean_test_" + score][i]
            self.cv_tab_eval = pd.concat([self.cv_tab_eval, pd.DataFrame([row])], ignore_index=True)


    def print_evaluate_crossval(self):
        for i in range(len(self.cv_comb_params)):
            row = {"hyperparameters": self.cv_comb_params[i]}
            for score in CV_SCORES[self.model_type]:
                row[score] = self.grid_search.cv_results_["mean_test_" + score]
            self.cv_tab_eval = pd.concat([self.cv_tab_eval, pd.DataFrame([row])], ignore_index=True)


    def conf_matrix(self):

        # Créez un graphique de la matrice de confusion
        fig = plt.figure(figsize=(6, 6))
        sns.heatmap(self.cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
        plt.xlabel('Valeurs Prédites')
        plt.ylabel('Valeurs Réelles')
        plt.title('Matrice de Confusion')
        plt.show()
        return st.pyplot(fig)


    def plot_reg(self):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.scatter(self.y_test, self.y_pred)
        line = mlines.Line2D([0, 1], [0, 1], color="green")
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlabel("true values")
        plt.ylabel("predicted values")
        plt.show()
        return st.pyplot(fig)


def create_tab_eval_reg():
    tab_eval = pd.DataFrame(columns=["hyperparameters", "rmse", "mae", "maxerror"])
    return tab_eval

def create_tab_eval_clf():
    tab_eval = pd.DataFrame(
        columns=["hyperparameters", "classe",
                 "accuracy", "precision", "recall", "f1-score"])
    return tab_eval

def create_tab_eval_crossval(scorings):
    tab_eval = pd.DataFrame(
        columns=["hyperparameters"] + scorings)
    return tab_eval
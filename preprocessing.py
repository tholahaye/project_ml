from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.figure_factory as ff
import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split


class MissingClassError(Exception):
    pass


class Preprocessing:

    def __init__(self, dataframe, model_type, choice_na, collinear_thresh, test_size=0.2, random_state=42):

        # TODO: Verifier l'utilite de garder l'original
        self.df_original = dataframe
        self.df = dataframe
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.choice_na = choice_na
        self.collinear_thresh = collinear_thresh

        self.remove_nan()

        if self.model_type == "Classification":
            self.classes_set = set(self.df['target'].unique())
        self.encoder()
        self.remove_outliers()

        self.X_df = self.df.drop(columns=['target'])
        self.y_df = self.df['target']

        self.scaler()

        # TODO: Attribuer correlation matrix
        #self.cr_matrix = self.correlation_matrix()

        try:
            self.X_train, self.X_test, \
                self.y_train, self.y_test = train_test_split(self.X_df, self.y_df,
                                                             test_size=self.test_size,
                                                             random_state=self.random_state)
        except MissingClassError:
            raise MissingClassError

    def remove_nan(self):
        if self.choice_na.lower() == 'median':
            for col in self.df.columns:
                if self.df[col].isna().any():
                    median_value = self.df[col].median()
                    self.df[col].fillna(median_value, inplace=True)
        elif self.choice_na.lower() == 'mean':
            for col in self.df.columns:
                if self.df[col].isna().any():
                    mean_value = self.df[col].mean()
                    self.df[col].fillna(mean_value, inplace=True)
        else:
            for col in self.df.columns:
                nan_count = self.df[col].isna().sum()
                rows_count = len(self.df)
                prop_nan = nan_count / rows_count
                if prop_nan > 0.4:
                    self.df.drop(col, axis=1, inplace=True)

    def encoder(self):
        le = LabelEncoder()
        for column in self.df.drop(columns=['target']).columns:
            if str(self.df.dtypes[column]) == 'object':
                le.fit(self.df[column].unique())
                self.df[column] = le.transform(self.df[column])

        # TODO: Elimination des outliers

    def remove_outliers(self):
        pass
        #z_sc = np.abs(stats.zscore(self.df))
        #self.df = self.df[(z_sc < 3).all(axis=1)]

    def scaler(self):
        scaler = StandardScaler()
        self.df = scaler.fit_transform(self.X_df, self.y_df)

    # TODO: Utile ?
    def outliers(self, contamination=0.05):
        outlier_detector = IsolationForest(contamination=contamination)
        outlier_labels = outlier_detector.fit_predict(self.X_df)
        self.df = self.df[outlier_labels == 1]
        return self.df

    def split_train_test(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X_df, self.y_df,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        if self.model_type == "Classification":
            classes = set(self.y_df.unique())
            classes_train = set(y_train.unique())
            if classes != classes_train:
                raise MissingClassError
        return x_train, x_test, y_train, y_test

    # TODO: Graph correlation matrix
    '''def correlation_matrix(self):
        corr_matrix = pd.DataFrame(self.df).corr()
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale='RdBu',
            annotation_text=corr_matrix.round(2).values,
            showscale=True,
            reversescale=True)

        fig.update_layout(
            title='Carte des corrélations',
            title_font_size=15,
            title_font=dict(weight='bold'),
            xaxis=dict(ticks='', side='top'),
            yaxis=dict(ticks='', side='left'))

        return fig'''
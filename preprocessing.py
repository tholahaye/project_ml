from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

        self.cr_matrix = self.corr_matrix()

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

    def drop_colinear(self):
        #TODO: Enlever les colinear
        pass

    def remove_outliers(self, threshold=1.5):

        numeric_columns = self.df.select_dtypes(include=['number']).columns
        for column in numeric_columns:
            if self.df[column].dtype != 'object':
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]

    def scaler(self):
        scaler = StandardScaler()
        self.df = scaler.fit_transform(self.X_df, self.y_df)

    def corr_matrix(self):
        corr = self.X_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig = plt.figure(figsize=(12, 10))
        cmap = 'coolwarm'
        sns.heatmap(corr, annot=True, cmap=cmap, mask=mask)
        plt.show()
        return fig

    def split_train_test(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X_df, self.y_df,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        if self.model_type == "Classification":
            classes = set(self.y_df.unique())
            #TODO: check for unbalanced dataset
            classes_train = set(y_train.unique())
            if classes != classes_train:
                raise MissingClassError
                #TODO check the distributions, not only the equality of sets
        return x_train, x_test, y_train, y_test

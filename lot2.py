from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

from sklearn.model_selection import train_test_split


class Preprocessing:

    def __init__(self, dataframe, test_size=0.2, random_state=42):

        # TODO: Verifier l'utilite de garder l'original
        self.df_original = dataframe
        self.df = dataframe
        self.test_size = test_size
        self.random_state = random_state

        self.remove_nan()
        self.encoder()
        self.remove_outliers()

        self.X_df = self.df.drop(columns=['target'])
        self.y_df = self.df['target']

        self.scaler()

        self.X_train, self.X_test, \
            self.y_train, self.y_test = train_test_split(self.X_df, self.y_df,
                                                         test_size=self.test_size,
                                                         random_state=self.random_state)

    # TODO: Gerer les NaN
    def remove_nan(self):

        self.df = self.df.dropna()

        # TODO: Encodage
    def encoder(self):
        le = LabelEncoder()
        for column in self.df.columns:
            if str(self.df.dtypes[column]) == 'object':
                le.fit(self.df[column].unique())
                self.df[column] = le.transform(self.df[column])

        # TODO: Elimination des outliers
    def remove_outliers(self):
        pass

    def scaler(self):
        pass  # TODO: A enlever dans le final
        scaler = StandardScaler()
        self.X_df = scaler.fit_transform(self.X_df, self.y_df)

    # TODO: Utile ?
    def outliers(self, contamination=0.05):
        outlier_detector = IsolationForest(contamination=contamination)
        outlier_labels = outlier_detector.fit_predict(self.X_df)
        self.df = self.df[outlier_labels == 1]
        return self.df



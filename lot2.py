from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
import pandas as pd 

from sklearn.model_selection import train_test_split


class Preprocessing:

    def __init__(self, dataframe, model_type, model, split_rate=0.2):
        self.df = dataframe
        self.df_original = dataframe
        self.type = model_type
        self.model = model
        self.split_rate = split_rate

        # TODO: Gerer les NaN
        self.df = self.df.dropna()

        # TODO: Encodage

        le = LabelEncoder()
        for column in self.df.column:
            if str(self.df.dtypes[column]) == 'object':
                le.fit(self.df[column].unique())
                self.df[column] = le.transform(self.df[column])

        #TODO: Elimination des outliers

        #TODO: Standardisation

        #TODO: Split du dataset en train/test
        X_df = self.df.drop(columns=['target'])
        y_df = self.df['target']

        self.X_train, self.X_test,\
            self.y_train, self.y_test = train_test_split(X_df, y_df,
                                                         test_size=self.split_rate,
                                                         random_state=1996)


def outliers(self, contamination=0.05):
    outlier_detector = IsolationForest(contamination=contamination)
    outlier_labels = outlier_detector.fit_predict(self.X)
    self.df = self.df[outlier_labels == 1]
    # Conservez les données originales
    self.original_df = self.df.copy()
    return self.df


'''size = df['target'].nunique()

if size == 2: 
    LE = LabelEncoder()
    df['target'] = LE.fit_transform(df['target'])
else:
    pass '''


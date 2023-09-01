from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd 

from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\digigi\Downloads\diabete.csv")

'''class ColumnCleaner:
    def __init__(self, data):
        self.data = df

    def drop_empty_columns(self):
        empty_cols = self.data.columns[self.data.isnull().all()]
        self.data.drop(empty_cols, axis=1, inplace=True)
        processor = ColumnCleaner(df)
        processor.drop_empty_columns()'''

df = df.drop('id', axis=1)
for colonne in df:
    df = df.dropna()

LE = LabelEncoder()

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



X = df
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1996)

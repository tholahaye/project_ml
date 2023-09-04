import psycopg2
import pandas as pd
import streamlit as st
from lot2 import Preprocessing
from constantes import structure


class AppWeb:

    def __init__(self):
        intro()
        st.title("Welcome")

        try:
            self.conn = psycopg2.connect(host="ec2-34-247-94-62.eu-west-1.compute.amazonaws.com",
                                         database="d4on6t2qk9dj5a",
                                         user="nxebpjsgxecqny",
                                         password="1da2f1f48e4a37bf64e3344fe7670a6547c169472263b62d042a01a8d08d2114")

            self.cursor = self.conn.cursor()

            self.table_names_list = self.get_table_names()

            st.sidebar.caption(":red[__Choose the parameters:__]")

            self.dataset_name = st.sidebar.selectbox("_Datasets:_", self.table_names_list)

            self.dataframe = self.df_creation()

            try:
                self.random_state = int(st.sidebar.text_input("Random state:",
                                                              help='Write an integer. It controls the '
                                                              'shuffling applied to the data before'
                                                              ' applying the split.'))
            except ValueError:
                st.sidebar.markdown(':red[__Error: The selected random state must be an integer__]')

            try:
                self.train_size = float(st.sidebar.text_input("Train size:",
                                                            help='Write a number between 0.0 and 1.0 and'
                                                                 ' represent the proportion of the dataset'
                                                                 ' to include in the train split.'))
                self.check_train_size()
            except ValueError:
                st.sidebar.markdown(":red[__Error: The train size must be a number between 0.0 and 1.0__]")

            self.model_type = self.get_model_type()

            preprocessing = Preprocessing(self.dataframe,
                                          train_size=self.train_size,
                                          random_state=self.random_state)
            # TODO/ A enlever du final
            st.dataframe(self.dataframe)

            self.dataframe = preprocessing.df
            # TODO: A enlever du final
            st.dataframe(self.dataframe)

            self.model = st.sidebar.selectbox("_Model:_", structure[self.model_type].keys())

            '''if self.model_type == 'Classification':
                self.model = st.sidebar.selectbox("_Model:_", ["ClassificationTree", "RandomForest"])
            if self.model_type == 'Regression':
                self.model = st.sidebar.selectbox("_Model:_", ["LinearRegression", "Ridge"])'''

            if self.model_type:
                # TODO: Prendre le dict d'hyperperametres
                self.hyperparameters_list = structure[self.model_type][self.model]['hyperparameters'].keys()
                # TODO: Mettre sous forme de dic {'nom': 'valeur'}
                self.hyperparameters_values = []

                for hp in self.hyperparameters_list:
                    # TODO: A appeler avec le modèle
                    hp = st.sidebar.text_input(f"Hyperparameter {hp}:", help='Hyperparameter descriptive')
                    self.hyperparameters_values.append(hp)

                if len(self.hyperparameters_values) == len(self.hyperparameters_list):
                    st.sidebar.text(self.hyperparameters_values)

        finally:
            self.conn.close()

    def get_table_names(self):
        self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        table_name_list = []
        for table in self.cursor.fetchall():
            table = str(table).strip("(),'")
            table_name_list.append(table)
        return table_name_list

    def df_creation(self):
        self.cursor.execute(f"SELECT * FROM {self.dataset_name}")
        data = self.cursor.fetchall()
        self.cursor.execute(f"SELECT * FROM information_schema.columns WHERE table_name = '{self.dataset_name}'")
        headers = []
        for element in self.cursor.fetchall():
            headers.append(element[3])
        df = pd.DataFrame(data=data, columns=headers)
        df.set_index('id', inplace=True)
        return df

    def check_train_size(self):
        if self.train_size < 0 or self.train_size > 1:
            raise ValueError

    def get_model_type(self):
        target_type = str(self.dataframe['target'].dtype)
        if target_type.startswith(('int', 'float')):
            return "Regression"
        else:
            return "Classification"


#TODO: A enlever avec l'instanciation du modele
'''def model():
    model_dic = {'Classification': {
                            "ClassificationTree": {'hyperparameters': ['a', 'b', 'c']},
                            "RandomForest": {'hyperparameters': ['d', 'e', 'f']}
                            },
                 'Regression': {
                    "LinearRegression": {'hyperparameters': ['a', 'b', 'c']},
                    "Ridge": {'hyperparameters': ['d', 'e', 'f']}
                            }
                 }
    return model_dic'''


def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )


App = AppWeb()

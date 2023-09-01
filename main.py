import psycopg2
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


class AppWeb:

    def __init__(self):

        intro()
        st.title("Welcome")

        self.conn = psycopg2.connect(host="ec2-34-247-94-62.eu-west-1.compute.amazonaws.com",
                                     database="d4on6t2qk9dj5a",
                                     user="nxebpjsgxecqny",
                                     password="1da2f1f48e4a37bf64e3344fe7670a6547c169472263b62d042a01a8d08d2114")

        self.cursor = self.conn.cursor()

        self.table_names_list = self.get_table_names()

        st.sidebar.caption(":red[__Choose the parameters:__]")

        self.dataset_name = st.sidebar.selectbox("_Datasets:_", self.table_names_list)

        self.dataframe = self.df_creation()

        self.get_model_type()

        self.type_model = self.get_model_type()

        if self.type_model == 'Classification':
            self.model = st.sidebar.selectbox("_Model:_", ["ClassificationTree", "RandomForest"])
        if self.type_model == 'Regression':
            self.model = st.sidebar.selectbox("_Model:_", ["LinearRegression", "Ridge"])

        self.hyperparameters_list = model()[self.type_model][self.model]['hyperparameters']

        self.hyperparameters_values = []

        for hp in self.hyperparameters_list:
            hp = st.sidebar.text_input(f"Hyperparameter {hp}:", help='Hyperparameter descriptive')# A appeler avec le modèle


            self.hyperparameters_values.append(hp)

        if len(self.hyperparameters_values) == len(self.hyperparameters_list):
            x = st.sidebar.expander
            with x:
                st.text(self.hyperparameters_values)

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
        st.dataframe(df)
        return df

    def get_model_type(self):
        target_type = str(self.dataframe['target'].dtype)
        if target_type.startswith(('int', 'float')):
            return "Regression"
        else:
            return "Classification"

# A enlever avec l'instanciation du modele
def model():
    model_dic = {'Classification': {
                            "ClassificationTree": {'hyperparameters': ['a', 'b', 'c']},
                            "RandomForest": {'hyperparameters': ['d', 'e', 'f']}
                            },
                 'Regression': {
                    "LinearRegression": {'hyperparameters': ['a', 'b', 'c']},
                    "Ridge": {'hyperparameters': ['d', 'e', 'f']}
                            }
                 }
    return model_dic

def intro():
    return st.set_page_config(
        page_title="ML Playground",
        layout="centered",
        initial_sidebar_state="auto"
    )


App = AppWeb()

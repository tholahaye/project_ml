import psycopg2
import pandas as pd
import streamlit as st
from lot2 import Preprocessing
from constantes import structure


class AppWeb:

    def __init__(self):
        st.set_page_config(
            page_title="ML Playground",
            layout="centered",
            initial_sidebar_state="auto"
        )

        st.title("Welcome")

        try:
            self.conn = psycopg2.connect(host="ec2-34-247-94-62.eu-west-1.compute.amazonaws.com",
                                         database="d4on6t2qk9dj5a",
                                         user="nxebpjsgxecqny",
                                         password="1da2f1f48e4a37bf64e3344fe7670a6547c169472263b62d042a01a8d08d2114")

            self.cursor = self.conn.cursor()

            self.table_names_list = self.get_table_names()

            st.sidebar.caption(":green[__Choose the parameters:__]")

            self.dataset_name = st.sidebar.selectbox("_Dataset:_", self.table_names_list)

            self.dataframe = self.df_creation()

            try:
                self.random_state = int(st.sidebar.number_input("Random state:",
                                                                min_value=0,
                                                                step=1,
                                                                help='Write an integer. It controls the '
                                                                'shuffling applied to the data before'
                                                                ' applying the split.'))
            except ValueError:
                st.sidebar.markdown(':red[__Error: The selected random state must be an integer__]')

            self.test_size = float(st.sidebar.number_input("Train size:",
                                                           min_value=0.01,
                                                           max_value=0.99,
                                                           help='Write a number between 0.0 and 1.0 and'
                                                           ' represent the proportion of the dataset'
                                                           ' to include in the test split.'))

            self.model_type = self.get_model_type()

            preprocessing = Preprocessing(self.dataframe,
                                          test_size=self.test_size,
                                          random_state=self.random_state)
            # TODO/ A enlever du final
            st.dataframe(self.dataframe)

            self.dataframe = preprocessing.df
            # TODO: A enlever du final
            st.dataframe(self.dataframe)

            self.model = st.sidebar.selectbox("_Model:_", structure[self.model_type].keys())

            if self.model_type:
                self.model_hyparameters = structure[self.model_type][self.model]['hyperparameters']
                self.hyperparameters_list = self.model_hyparameters.keys()
                self.hyperparameters_values = dict()

                for hp in self.hyperparameters_list:
                    if self.model_hyparameters[hp]['type'] == 'str':
                        hp_value = st.sidebar.selectbox(f"Hyperparameter {hp}:",
                                                        self.model_hyparameters[hp]['values'],
                                                        help=f"{self.model_hyparameters[hp]['description']}")
                    if self.model_hyparameters[hp]['type'] in ['int', 'float']:
                        if self.model_hyparameters[hp]['max_value'] != float('inf'):
                            hp_value = st.sidebar.number_input(label=f"Hyperparameter {hp}:",
                                                               min_value=self.model_hyparameters[hp]['min_value'],
                                                               max_value=self.model_hyparameters[hp]['max_value'],
                                                               help=f"{self.model_hyparameters[hp]['description']}")
                        else:
                            hp_value = st.sidebar.number_input(label=f"Hyperparameter {hp}:",
                                                               min_value=self.model_hyparameters[hp]['min_value'],
                                                               help=f"{self.model_hyparameters[hp]['description']}")

                    if hp_value:
                        self.hyperparameters_values[hp] = hp_value

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

    def get_model_type(self):
        target_type = str(self.dataframe['target'].dtype)
        if target_type.startswith(('int', 'float')):
            return "Regression"
        else:
            return "Classification"



if __name__ == '__main__':
    AppWeb()

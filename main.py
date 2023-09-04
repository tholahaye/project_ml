import psycopg2
import pandas as pd
import streamlit as st
from lot2 import Preprocessing, MissingClassError
from ml import MachineLearning
from constantes import STRUCTURE


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

            with st.sidebar.expander(':blue[__Parameters__]'):
                try:
                    self.random_state = int(st.number_input("Random state:",
                                                            value=42,
                                                            min_value=0,
                                                            step=1,
                                                            help='Write an integer. It controls the '
                                                            'shuffling applied to the data before'
                                                            ' applying the split.'))
                except ValueError:
                    st.markdown(':red[__Error: The selected random state must be an integer__]')

                self.test_size = float(st.number_input("Test size:",
                                                       value=0.2,
                                                       min_value=0.01,
                                                       max_value=0.99,
                                                       help='Write a number between 0.0 and 1.0 and'
                                                       ' represent the proportion of the dataset'
                                                       ' to include in the test split.'))

                self.model_type = self.get_model_type()

            try:
                preprocessing = Preprocessing(self.dataframe,
                                              model_type= self.model_type,
                                              test_size=self.test_size,
                                              random_state=self.random_state)
                self.X_train = preprocessing.X_train
                self.X_test =  preprocessing.X_test
                self.y_train = preprocessing.y_train
                self.y_test = preprocessing.y_test
                # TODO: A enlever du final
                st.dataframe(self.dataframe)

                self.classes_set = preprocessing.classes_set  # TODO: Utiliser dans la creation du modele

                self.dataframe = preprocessing.df
                # TODO: A enlever du final
                st.dataframe(self.dataframe)

                self.model = st.sidebar.selectbox("_Model:_", STRUCTURE[self.model_type].keys())

                if self.model_type:
                    self.model_hyparameters = STRUCTURE[self.model_type][self.model]['hyperparameters']
                    self.hyperparameters_list = self.model_hyparameters.keys()
                    self.hyperparameters_values = dict()

                    self.hyperparameters()

                    if len(self.hyperparameters_values) == len(self.hyperparameters_list):
                        st.sidebar.text(self.hyperparameters_values)

            except MissingClassError:
                st.markdown(":red[__Missing class in the training values. Please change your random state.__]")

            self.ml = MachineLearning(model_type= self.model_type,
                            model_name=self.model,
                            hyper_params=self.hyperparameters_values,
                            X_train=self.X_train,
                            X_test=self.X_test,
                            y_train=self.y_train,
                            y_test=self.y_test,
                            classes=self.classes_set)
            st.dataframe(self.ml.tab_eval)

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

    def hyperparameters(self):
        with st.sidebar.expander(":blue[__Hyperparameters__]"):
            for hp in self.hyperparameters_list:
                if self.model_hyparameters[hp]['type'] == 'str':
                    hp_value = st.selectbox(f"Hyperparameter {hp}:",
                                            self.model_hyparameters[hp]['values'],
                                            help=f"{self.model_hyparameters[hp]['description']}")
                if self.model_hyparameters[hp]['type'] in ['int', 'float']:
                    if self.model_hyparameters[hp]['optional']:
                        hp_show = st.checkbox(label=f"Hyperparameter {hp}:",
                                              value=False,
                                              help=f"{self.model_hyparameters[hp]['description']}")
                        if hp_show:
                            if self.model_hyparameters[hp]['max_value'] != float('inf'):
                                hp_value = st.number_input(label=f"Value {hp}:",
                                                           min_value=self.model_hyparameters[hp]['min_value'],
                                                           max_value=self.model_hyparameters[hp]['max_value'],)
                            else:
                                hp_value = st.number_input(label=f"Value {hp}:",
                                                           min_value=self.model_hyparameters[hp]['min_value'])

                    else:
                        if self.model_hyparameters[hp]['max_value'] != float('inf'):
                            hp_value = st.number_input(label=f"Hyperparameter {hp}:",
                                                       value=self.model_hyparameters[hp]['default'],
                                                       min_value=self.model_hyparameters[hp]['min_value'],
                                                       max_value=self.model_hyparameters[hp]['max_value'],
                                                       help=f"{self.model_hyparameters[hp]['description']}")
                        else:
                            hp_value = st.number_input(label=f"Hyperparameter {hp}:",
                                                       value=self.model_hyparameters[hp]['default'],
                                                       min_value=self.model_hyparameters[hp]['min_value'],
                                                       help=f"{self.model_hyparameters[hp]['description']}")

                self.hyperparameters_values[hp] = hp_value


if __name__ == '__main__':
    AppWeb()

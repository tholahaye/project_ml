import psycopg2
import pandas as pd
import streamlit as st
from preprocessing import Preprocessing, MissingClassError
import ml
from constantes import STRUCTURE, CV_SCORES, CV_MAX_RES
from decouple import config


class AppWeb:

    def __init__(self):
        st.set_page_config(
            page_title="ML Playground",
            layout="wide",
            initial_sidebar_state="auto"
        )

        st.title("Bienvenue chez les dauphins de Chine !")

        try:

            self.conn = psycopg2.connect(host=config('HOST'),
                                         database=config('DATABASE'),
                                         user=config('USER'),
                                         password=config('PASSWORD'))

            self.cursor = self.conn.cursor()

            self.table_names_list = self.get_table_names()

            st.sidebar.caption(":green[__Choose the parameters:__]")

            self.dataset_name = st.sidebar.selectbox("_Dataset:_", self.table_names_list)

            self.dataframe = self.df_creation()

            with st.sidebar.expander(':blue[__Preprocessing parameters__]'):
                try:
                    self.random_state = int(st.number_input("Random state:",
                                                            value=42,
                                                            min_value=0,
                                                            step=1,
                                                            help='Choose or write an integer. It controls the '
                                                                 'shuffling applied to the data before'
                                                                 ' applying the split.'))
                except ValueError:
                    st.markdown(':red[__Error: The selected random state must be an integer__]')

                self.test_size = float(st.number_input("Test size:",
                                                       value=0.2,
                                                       min_value=0.01,
                                                       max_value=0.99,
                                                       help='Choose or write a number between 0.0 and 1.0 and'
                                                            ' represent the proportion of the dataset'
                                                            ' to include in the test split.'))

                self.choice_na = st.selectbox("NaN treatment:",
                                              ['Remove line', 'Replaced by mean', 'Replaced by median'],
                                              help="Choose how the missing values will be treated.")

                self.collinear_thresh = float(st.number_input("Collinearity threshold:",
                                                              value=0.5,
                                                              min_value=0.01,
                                                              max_value=0.99,
                                                              help='Choose or write a number between 0.0 and 1.0 and'
                                                                   'represent the threshold above which'
                                                                   'variables are considered collinear.'))

                self.model_type = self.get_model_type()

            try:

                # Preprocessing **************************************************************************************
                preprocessing = Preprocessing(self.dataframe,
                                              model_type=self.model_type,
                                              test_size=self.test_size,
                                              random_state=self.random_state,
                                              choice_na=self.choice_na,
                                              collinear_thresh=self.collinear_thresh)
                self.X_train = preprocessing.X_train
                self.X_test = preprocessing.X_test
                self.y_train = preprocessing.y_train
                self.y_test = preprocessing.y_test
                self.classes_set = set()

                with st.expander(":green[__Original dataframe__]"):
                    st.dataframe(self.dataframe)
                    st.pyplot(preprocessing.cr_matrix)

                self.dataframe = preprocessing.df

                if self.model_type == "Classification":
                    self.classes_set = preprocessing.classes_set

                with st.expander(":green[__Processed dataframe__]"):
                    st.dataframe(self.dataframe)

                with st.expander(":green[__Train/test__]"):
                    st.dataframe(preprocessing.X_train)
                    st.dataframe(preprocessing.X_test)
                    st.dataframe(preprocessing.y_train)
                    st.dataframe(preprocessing.y_test)

                self.model = st.sidebar.selectbox("_Model:_", STRUCTURE[self.model_type].keys())
                if len(STRUCTURE[self.model_type][self.model]['hyperparameters']) != 0:
                    self.cross_val = st.sidebar.toggle("Compare several parameters configurations"
                                                       " with Cross-validation",
                                                       help='Help cross-validation')  # TODO: Help cross_val
                else:
                    self.cross_val = False

                if self.cross_val:
                    with st.sidebar.expander(':blue[__Cross-validation parameters__]'):
                        try:
                            self.nfold = int(st.number_input("Fold number:",
                                                                    value=5,
                                                                    min_value=2,
                                                                    step=1,
                                                                    # TODO: A completer
                                                                    help='Choose or write an integer. It controls the '
                                                                         'number of folds'
                                                                         ' applied to the model.'))
                        except ValueError:
                            st.markdown(':red[__Error: The number of fold must be an integer__]')

                        self.cv_score = st.selectbox('Optimize: ',
                                                     CV_SCORES[self.model_type],
                                                     help='The score to optimize during the cross-validation')


                if self.model_type:
                    self.model_hyperparameters = STRUCTURE[self.model_type][self.model]['hyperparameters']
                    self.hyperparameters_list = self.model_hyperparameters.keys()
                    self.hyperparameters_values = dict()
                    if self.cross_val:
                        self.hyperparameter_setting_crossval()
                    else:
                        self.hyperparameter_setting()

            except MissingClassError:
                st.markdown(":red[__Training data do not have all classes"
                            " represented, please change your random seed.__]")

            #  Machine learning *******************************************************************************
            try:
                self.ml = ml.MachineLearning(model_type=self.model_type,
                                             model_name=self.model,
                                             hyper_params=self.hyperparameters_values,
                                             X_train=self.X_train,
                                             X_test=self.X_test,
                                             y_train=self.y_train,
                                             y_test=self.y_test,
                                             classes=self.classes_set,
                                             cross_val=self.cross_val,
                                             cv_nfold=self.nfold,
                                             cv_score="accuracy")

                if self.cross_val:
                    with st.expander("Parameters' selection with cross validation"):
                        st.dataframe(self.ml.cv_tab_eval.sort_values(by=self.cv_score, ascending=False)[range(CV_MAX_RES),:])

                with st.expander(":green[__Evaluation__]"):
                    st.dataframe(self.ml.tab_eval)
                
                    if self.model_type == 'Classification':
                        pass
                        st.markdown(":green[__Confusion Matrix__]")
                        self.ml.conf_matrix()


            except AttributeError:
                # TODO: Compléter le rapport d'erreur
                st.markdown(":red[__AttributeError: Oopsie!__] :see_no_evil:")

        finally:
            if self.conn:
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

    def hyperparameter_setting_crossval(self):
        with st.sidebar.expander(":blue[__Hyperparameters__]"):
            for hp in self.hyperparameters_list:
                if self.model_hyperparameters[hp]['type'] == 'str':
                    hp_value = st.multiselect(f"Hyperparameter {hp}:",
                                              self.model_hyperparameters[hp]['values'],
                                              help=f"{self.model_hyperparameters[hp]['description']}")
                if self.model_hyperparameters[hp]['type'] in ['int', 'float']:
                    if self.model_hyperparameters[hp]['optional']:
                        hp_show = st.checkbox(label=f"Hyperparameter {hp}:",
                                              value=False,
                                              help=f"{self.model_hyperparameters[hp]['description']}")

                    if not self.model_hyperparameters[hp]['optional'] or hp_show:
                        hp_value = st.text_input(label=f"Hyperparameter {hp}:",
                                                 value=self.model_hyperparameters[hp]['default'],
                                                 help=f"{self.model_hyperparameters[hp]['description']}."
                                                      "Separate the wanted values by ';'.")

                        hp_value = hp_value.split(';')
                        for value in hp_value:
                            value = str(value)
                            value = value.strip()
                            try:
                                if value == '' and len(hp_value) != 0:
                                    continue

                                hp_type = self.model_hyperparameters[hp]['type']
                                try:
                                    if hp_type == 'int':
                                        value = int(value)
                                    elif hp_type == 'float':
                                        value = float(value)
                                except ValueError:
                                    st.markdown(f":red[__Error: You must "
                                                f"type a list of {hp_type} separated by ';'.__]")

                                try:
                                    min_hp = self.model_hyperparameters[hp]['min_value']
                                    if min_hp > value:
                                        raise InferiorToMinError
                                except KeyError:
                                    pass
                                except TypeError:
                                    pass

                                try:
                                    max_hp = self.model_hyperparameters[hp]['max_value']
                                    if max_hp < value:
                                        raise SuperiorToMaxError

                                except KeyError:
                                    pass
                                except TypeError:
                                    pass

                                hp_value.append(value)

                            except InferiorToMinError:
                                st.markdown(f":red[__Error: Your values must be superior or equal to {min_hp}.__]")
                            except SuperiorToMaxError:
                                st.markdown(f":red[__Error: Your values must be inferior or equal to {max_hp}.__]")

                        hp_value = list(filter(None, hp_value))
                if len(hp_value) == 0:
                    st.markdown(":red[__Error: You must enter at least one value.__]")
                else:
                    self.hyperparameters_values[hp] = hp_value

    def hyperparameter_setting(self):
        with st.sidebar.expander(":blue[__Hyperparameters__]"):
            for hp in self.hyperparameters_list:
                if self.model_hyperparameters[hp]['type'] == 'str':
                    hp_value = st.selectbox(f"Hyperparameter {hp}:",
                                            self.model_hyperparameters[hp]['values'],
                                            help=f"{self.model_hyperparameters[hp]['description']}")
                if self.model_hyperparameters[hp]['type'] in ['int', 'float']:

                    if self.model_hyperparameters[hp]['optional']:
                        hp_show = st.checkbox(label=f"Hyperparameter {hp}:",
                                              value=False,
                                              help=f"{self.model_hyperparameters[hp]['description']}")
                        if not hp_show:
                            hp_value = None

                    if not self.model_hyperparameters[hp]['optional'] or hp_show:
                        hp_value = self.hyperparameter_number_input(hp)

                self.hyperparameters_values[hp] = hp_value

    def hyperparameter_number_input(self, hp):
        hp_type = self.model_hyperparameters[hp]['type']
        min_value = self.model_hyperparameters[hp]['min_value']
        max_value = self.model_hyperparameters[hp]['max_value']
        value = self.model_hyperparameters[hp]['default']
        if max_value == float('inf'):
            max_value = None

        if hp_type == 'int':
            try:
                value = int(value)
            except ValueError:
                value = None
            step = int(1)

        if hp_type == 'float':
            min_value = float(min_value)
            if max_value:
                max_value = float(max_value)
            try:
                value = float(value)
            except ValueError:
                value = None
            step = 0.01

        hp_value = st.number_input(label=f"Hyperparameter {hp}:",
                                   value=value,
                                   step=step,
                                   min_value=min_value,
                                   max_value=max_value,
                                   help=f"{self.model_hyperparameters[hp]['description']}")

        return hp_value




class InferiorToMinError(Exception):
    pass


class SuperiorToMaxError(Exception):
    pass


if __name__ == '__main__':
    AppWeb()

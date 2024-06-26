'''
ML model that, given a COVID-19 patient's current
symptom, status, and medical history, will predict whether the patient is at high
risk or not.

Brydon Wall
BNFOthon 2024

https://keras.io/examples/structured_data/structured_data_classification_from_scratch/
'''
import numpy as np
import pandas as pd
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import pandas as pd
import keras
from keras import layers
import pickle
import argparse

#import keras

class COVID:

    covid_columns = [
        'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED',
        'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
        'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
        'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
    ]

    def __init__(
            self,
            in_path: str = os.path.normpath('./Covid Data.csv'),
            processed_path: str = './Processed Covid Data.csv',
            class_col: str = 'PATIENT_TYPE',
            seed: int = 42,
            validation_frac = 0.2,
            exclude = [
                'USMER', 'MEDICAL_UNIT', 'CLASIFFICATION_FINAL', 'ICU'
            ],
            batch = 32,
            only_covid_positive: bool = False
        ) -> None:
        self.in_path = in_path
        self.proc_path = processed_path
        self.target = class_col
        self._data = None
        self._train = None
        self._validation = None
        self.seed = seed
        self.validation_frac = validation_frac
        self.exclude = exclude
        self.batch = 32
        self._train_ds = None
        self._val_ds = None
        self._features = None,
        self.only_covid_positive = only_covid_positive

    def __process_csv(self, out_path: str = 'Processed Covid Data.csv') -> pd.DataFrame:

        covid_df = pd.read_csv(self.in_path)

        #covid_df.replace(97, np.nan, inplace=True)

        convert_list = [
            'USMER', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'DIABETES',
            'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE',
            'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU'
        ]

        # Reformat data to 0 and 1 for classed columns
        covid_df = covid_df.replace(np.nan, 97)
        covid_df[convert_list] = covid_df[convert_list].map(lambda x: {1: 0, 2: 1}.get(x, 97))
        covid_df['DATE_DIED'] = covid_df['DATE_DIED'].map(lambda x: 1 if x == '9999-99-99' else 0)
        covid_df['PREGNANT'] = covid_df['PREGNANT'].map({1: 0, 2: 1, 97: 97})
        covid_df['CLASIFFICATION_FINAL'] = covid_df['CLASIFFICATION_FINAL'].map(lambda x: 1 if x <= 3 else 0)

        covid_df.loc[covid_df['PATIENT_TYPE'] == 0, 'ICU'] = covid_df.loc[covid_df['PATIENT_TYPE'] == 0, 'ICU'].replace(97, 0)
        covid_df.loc[covid_df['SEX'] == 1, 'PREGNANT'] = covid_df.loc[covid_df['SEX'] == 1, 'PREGNANT'].replace(97, 0)
        
        covid_df.to_csv(out_path)

        return covid_df

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            if not os.path.isfile(self.proc_path):
                self._data = self.__process_csv(self.proc_path)
            else:
                self._data = pd.read_csv(self.proc_path, index_col=0)

            # Drop columns / subset
            if self.only_covid_positive:
                self._data = self._data[self._data['CLASIFFICATION_FINAL'] == 1]
            self._data = self._data.drop(columns=self.exclude)

            # Subset so that p_types are equal
            p_type_0 = len(self._data[self._data['PATIENT_TYPE'] == 0])
            p_type_1 = len(self._data[self._data['PATIENT_TYPE'] == 1])
            p_type_frac = p_type_1 / p_type_0
            self._data = pd.concat([self._data[self._data['PATIENT_TYPE'] == 0].sample(
                frac=p_type_frac, random_state=self.seed
            ), self._data[self._data['PATIENT_TYPE'] == 1]])

        return self._data

    @property
    def train(self) -> pd.DataFrame:
        if self._train is None:
            self._train = self.data.drop(self.validation.index)
        return self._train

    @property
    def validation(self) -> pd.DataFrame:
        if self._validation is None:
            self._validation = self.data.sample(
                frac=self.validation_frac, random_state=self.seed
            )
        return self._validation
        
    def _get_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        dataframe = df.copy()
        labels = dataframe.pop(self.target)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(dataframe))

        return ds

    @property
    def train_ds(self):
        if self._train_ds is None:
            train_ds = self._get_dataset(self.train)
            self._train_ds = train_ds.batch(self.batch)
        return self._train_ds

    @property
    def val_ds(self):
        if self._val_ds is None:
            val_ds = self._get_dataset(self.validation)
            self._val_ds = val_ds.batch(self.batch)
        return self._val_ds
            

    def train_model(self, path = './model_1.keras') -> keras.Model:
                
        train_ds = self.train_ds
        val_ds = self.val_ds
        #all_f_layers_pkl = './temp_f_layers.pkl'
        #if not os.path.isfile(all_f_layers_pkl):

        # Categorical features encoded as integers
        #usmer = keras.Input(shape=(1,), name="USMER", dtype="int64")
        #medical_unit = keras.Input(shape=(1,), name="MEDICAL_UNIT", dtype="int64")
        sex = keras.Input(shape=(1,), name="SEX", dtype="int64")
        #patient_type = keras.Input(shape=(1,), name="PATIENT_TYPE", dtype="int64")
        #date_died = keras.Input(shape=(1,), name="DATE_DIED", dtype="int64")
        #intubed = keras.Input(shape=(1,), name="INTUBED", dtype="int64")
        pneumonia = keras.Input(shape=(1,), name="PNEUMONIA", dtype="int64")
        pregnant = keras.Input(shape=(1,), name="PREGNANT", dtype="int64")
        diabetes = keras.Input(shape=(1,), name="DIABETES", dtype="int64")
        copd = keras.Input(shape=(1,), name="COPD", dtype="int64")
        asthma = keras.Input(shape=(1,), name="ASTHMA", dtype="int64")
        inmsupr = keras.Input(shape=(1,), name="INMSUPR", dtype="int64")
        hipertension = keras.Input(shape=(1,), name="HIPERTENSION", dtype="int64")
        other_disease = keras.Input(shape=(1,), name="OTHER_DISEASE", dtype="int64")
        cardiovascular = keras.Input(shape=(1,), name="CARDIOVASCULAR", dtype="int64")
        obesity = keras.Input(shape=(1,), name="OBESITY", dtype="int64")
        renal_chronic = keras.Input(shape=(1,), name="RENAL_CHRONIC", dtype="int64")
        tobacco = keras.Input(shape=(1,), name="TOBACCO", dtype="int64")
        #clasiffication_final = keras.Input(shape=(1,), name="CLASIFFICATION_FINAL", dtype="int64")
        #icu = keras.Input(shape=(1,), name="ICU", dtype="int64")

        # Numerical features
        age = keras.Input(shape=(1,), name="AGE")

        all_inputs = [
            #usmer,
            #medical_unit,
            sex,
            #patient_type,
            #date_died,
            #intubed,
            pneumonia,
            age,
            pregnant,
            diabetes,
            copd,
            asthma,
            inmsupr,
            hipertension,
            other_disease,
            cardiovascular,
            obesity,
            renal_chronic,
            tobacco#,
            #clasiffication_final,
            #icu
        ]


        # Integer categorical features
        #usmer_encoded = encode_categorical_feature(sex, "USMER", train_ds, False)
        #medical_unit_encoded = encode_categorical_feature(sex, "MEDICAL_UNIT", train_ds, False)
        sex_encoded = encode_categorical_feature(sex, "SEX", train_ds, False)
        #patient_type_encoded = encode_categorical_feature(sex, "PATIENT_TYPE", train_ds, False)
        #date_died_encoded = encode_categorical_feature(sex, "DATE_DIED", train_ds, False)
        #intubed_encoded = encode_categorical_feature(sex, "INTUBED", train_ds, False)
        pneumonia_encoded = encode_categorical_feature(sex, "PNEUMONIA", train_ds, False)
        pregnant_encoded = encode_categorical_feature(sex, "PREGNANT", train_ds, False)
        diabetes_encoded = encode_categorical_feature(sex, "DIABETES", train_ds, False)
        copd_encoded = encode_categorical_feature(sex, "COPD", train_ds, False)
        asthma_encoded = encode_categorical_feature(sex, "ASTHMA", train_ds, False)
        inmsupr_encoded = encode_categorical_feature(sex, "INMSUPR", train_ds, False)
        hipertension_encoded = encode_categorical_feature(sex, "HIPERTENSION", train_ds, False)
        other_disease_encoded = encode_categorical_feature(sex, "OTHER_DISEASE", train_ds, False)
        cardiovascular_encoded = encode_categorical_feature(sex, "CARDIOVASCULAR", train_ds, False)
        obesity_encoded = encode_categorical_feature(sex, "OBESITY", train_ds, False)
        renal_chronic_encoded = encode_categorical_feature(sex, "RENAL_CHRONIC", train_ds, False)
        tobacco_encoded = encode_categorical_feature(sex, "TOBACCO", train_ds, False)
        #clasiffication_final_encoded = encode_categorical_feature(sex, "CLASIFFICATION_FINAL", train_ds, False)
        #icu_encoded = encode_categorical_feature(sex, "ICU", train_ds, False)

        # Numerical features
        age_encoded = age #encode_numerical_feature(age, "age", train_ds)

        all_f_layers = [
            #usmer_encoded,
            #medical_unit_encoded,
            sex_encoded,
            #patient_type_encoded,
            #date_died_encoded,
            #intubed_encoded,
            pneumonia_encoded,
            age_encoded,
            pregnant_encoded,
            diabetes_encoded,
            copd_encoded,
            asthma_encoded,
            inmsupr_encoded,
            hipertension_encoded,
            other_disease_encoded,
            cardiovascular_encoded,
            obesity_encoded,
            renal_chronic_encoded,
            tobacco_encoded,
            #clasiffication_final_encoded,
            #icu_encoded
        ]

        '''
            with open(all_f_layers_pkl, 'wb') as out_file:
                pickle.dump(all_f_layers, out_file)

        else:
            with open(all_f_layers_pkl, 'rb') as in_file:
                all_f_layers = pickle.load(in_file)'''

        all_features = layers.concatenate(all_f_layers)

        x = layers.Dense(self.batch, activation="relu")(all_features)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(all_inputs, output)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-5),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )
        model.fit(train_ds, epochs=20, validation_data=val_ds)
    
        return model

    @property
    def features(self):
        if not self._features:
            self._features = self.train_model()
        return self._features

    def run(self, args):

        model = self.train_model()

        sample = {
            'SEX': args.SEX,
            #'DATE_DIED': 1,
            #'INTUBED': 0,
            'PNEUMONIA': args.PNEUMONIA,
            'AGE': args.AGE,
            'PREGNANT': args.PREGNANT,
            'DIABETES': args.DIABETES,
            'COPD': args.COPD,
            'ASTHMA': args.ASTHMA,
            'INMSUPR': args.INMSUPR,
            'HIPERTENSION': args.HIPERTENSION,
            'OTHER_DISEASE': args.OTHER_DISEASE,
            'CARDIOVASCULAR': args.CARDIOVASCULAR,
            'OBESITY': args.OBESITY,
            'RENAL_CHRONIC': args.RENAL_CHRONIC,
            'TOBACCO': args.TOBACCO
        }


        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        predictions = model.predict(input_dict)
        print(
            f"This patient's risk factor for severe COVID-19 is {100 * predictions[0][0]:.1f}% as evaluated by our model."
        )

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization()

    def f_1(x, y):
        return x[name]
    
    def f_2(x):
        tf.expand_dims(x, -1)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = layers.StringLookup if is_string else layers.IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    def f_1(x, y):
        return x[name]
    
    def f_2(x):
        tf.expand_dims(x, -1)

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def parse_arguments():
    parser = argparse.ArgumentParser(description="Neural network COVID-19 severity classifier")

    # Required argument
    parser.add_argument('--in_path', type=str, default=os.path.normpath('./Covid Data.csv'),
                        help='Input path for the data (./Covid Data.csv from https://www.kaggle.com/datasets/meirnizri/covid19-dataset)')

    # Optional arguments with default values
    parser.add_argument('--processed_path', type=str, default=os.path.normpath('./Processed Covid Data.csv'),
                        help='Path for the processed data file to be created')
    parser.add_argument('--class_col', type=str, default='PATIENT_TYPE',
                        help='Column name for the class label - not implemented / do not change')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--validation_frac', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--exclude', nargs='+', default=['USMER', 'MEDICAL_UNIT', 'CLASIFFICATION_FINAL', 'ICU'],
                        help='List of columns to exclude from the data, defaults: USMER MEDICAL_UNIT CLASIFFICATION_FINAL ICU')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size for data processing (integer)')
    parser.add_argument('--only_positive', type=bool, default=False,
                        help='Subset the data to only patients who test positive for COVID-19 (True or False)')

    # Default values for additional parameters
    parser.add_argument('--SEX', type=int, default=0,
                        help='0 for female and 1 for male.')
    parser.add_argument('--PNEUMONIA', type=int, default=0,
                        help='whether the patient already have air sacs inflammation or not. (0 or 1)')
    parser.add_argument('--AGE', type=int, default=35,
                        help='integer age of the patient.')
    parser.add_argument('--PREGNANT', type=int, default=1,
                        help='whether the patient is pregnant or not. (0 or 1)')
    parser.add_argument('--DIABETES', type=int, default=0,
                        help='whether the patient has diabetes or not. (0 or 1)')
    parser.add_argument('--COPD', type=int, default=0,
                        help='Indicates whether the patient has Chronic obstructive pulmonary disease or not. (0 or 1)')
    parser.add_argument('--ASTHMA', type=int, default=1,
                        help='whether the patient has asthma or not. (0 or 1)')
    parser.add_argument('--INMSUPR', type=int, default=1,
                        help='whether the patient is immunosuppressed or not. (0 or 1)')
    parser.add_argument('--HIPERTENSION', type=int, default=1,
                        help='whether the patient has hypertension or not. (0 or 1)')
    parser.add_argument('--OTHER_DISEASE', type=int, default=1,
                        help='whether the patient has other disease or not. (0 or 1)')
    parser.add_argument('--CARDIOVASCULAR', type=int, default=1,
                        help='whether the patient has heart or blood vessels related disease. (0 or 1)')
    parser.add_argument('--OBESITY', type=int, default=1,
                        help='whether the patient is obese or not. (0 or 1)')
    parser.add_argument('--RENAL_CHRONIC', type=int, default=1,
                        help='whether the patient has chronic renal disease or not. (0 or 1)')
    parser.add_argument('--TOBACCO', type=int, default=1,
                        help='whether the patient is a tobacco user. (0 or 1)')

    return parser.parse_args()

def main():
    args = parse_arguments()
    covid = COVID(
        in_path=args.in_path,
        processed_path=args.processed_path,
        class_col=args.class_col,
        seed=args.seed,
        validation_frac=args.validation_frac,
        exclude=args.exclude,
        batch=args.batch,
        only_covid_positive = args.only_positive
    )

    '''
    Namespace(in_path='./Covid Data.csv', processed_path='./Processed Covid Data.csv', class_col='PATIENT_TYPE', seed=42, validation_frac=0.2, exclude=['USMER', 'MEDICAL_UNIT', 'CLASIFFICATION_FINAL', 'ICU'], batch=32, SEX=0, PNEUMONIA=0, AGE=35, PREGNANT=1, DIABETES=0, COPD=0, ASTHMA=1, INMSUPR=1, HIPERTENSION=1, OTHER_DISEASE=1, CARDIOVASCULAR=1, OBESITY=1, RENAL_CHRONIC=1, TOBACCO=1)
    '''
    #print(covid.data)
    covid.run(args)

if __name__ == '__main__':
    main()

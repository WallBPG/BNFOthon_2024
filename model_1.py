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
            in_path: str,
            processed_path: str = './Processed Covid Data',
            class_col: str = 'PATIENT_TYPE',
            seed: int = 42,
            validation_frac = 0.2,
            exclude = [
                'USMER', 'MEDICAL_UNIT', 'CLASIFFICATION_FINAL', 'ICU'
            ],
            batch = 32
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

    def __process_csv(self, out_path: str = 'Processed Covid Data') -> pd.DataFrame:
        '''
        - sex: 1 for female and 2 for male.
        - age: of the patient.
        - classification: covid test findings. Values 1-3 mean that the patient was diagnosed with covid in different
        degrees. 4 or higher means that the patient is not a carrier of covid or that the test is inconclusive.
        - patient type: type of care the patient received in the unit. 1 for returned home and 2 for hospitalization.
        - pneumonia: whether the patient already have air sacs inflammation or not.
        - pregnancy: whether the patient is pregnant or not.
        - diabetes: whether the patient has diabetes or not.
        - copd: Indicates whether the patient has Chronic obstructive pulmonary disease or not.
        - asthma: whether the patient has asthma or not.
        - inmsupr: whether the patient is immunosuppressed or not.
        - hypertension: whether the patient has hypertension or not.
        - cardiovascular: whether the patient has heart or blood vessels related disease.
        - renal chronic: whether the patient has chronic renal disease or not.
        - other disease: whether the patient has other disease or not.
        - obesity: whether the patient is obese or not.
        - tobacco: whether the patient is a tobacco user.
        - usmr: Indicates whether the patient treated medical units of the first, second or third level.
        - medical unit: type of institution of the National Health System that provided the care.
        - intubed: whether the patient was connected to the ventilator.
        - icu: Indicates whether the patient had been admitted to an Intensive Care Unit.
        - date died: If the patient died indicate the date of death, and 9999-99-99 otherwise.
        '''

        covid_df = pd.read_csv(self.in_path)

        covid_df.replace(97, np.nan, inplace=True)

        convert_list = [
            'USMER', 'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'DIABETES',
            'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE',
            'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU'
        ]

        # Reformat data to 0 and 1 for classed columns
        covid_df[convert_list] = covid_df[convert_list].map(lambda x: {1: 0, 2: 1}.get(x, np.nan))
        covid_df['DATE_DIED'] = covid_df['DATE_DIED'].map(lambda x: 1 if x == '9999-99-99' else 0)
        covid_df['PREGNANT'] = covid_df['PREGNANT'].map({1: 0, 2: 1, np.nan: 0})
        covid_df['CLASIFFICATION_FINAL'] = covid_df['CLASIFFICATION_FINAL'].map(lambda x: 1 if x <= 3 else 0)

        covid_df.to_csv(out_path)
        return covid_df

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            if not os.path.isfile(self.proc_path):
                self._data = self.__process_csv(self.proc_path)
            else:
                self._data = pd.read_csv(self.proc_path, index_col=0)
            
            self._data = self._data.drop(columns=self.exclude)
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

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = layers.Normalization()

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

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


def main():
    covid = COVID('./Covid Data.csv')

    val_dataframe = covid.validation
    train_dataframe = covid.train

    train_ds = covid._get_dataset(train_dataframe)
    val_ds = covid._get_dataset(val_dataframe)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    # Categorical features encoded as integers
    usmer = keras.Input(shape=(1,), name="usmer", dtype="int64")
    medical_unit = keras.Input(shape=(1,), name="medical_unit", dtype="int64")
    sex = keras.Input(shape=(1,), name="sex", dtype="int64")
    patient_type = keras.Input(shape=(1,), name="patient_type", dtype="int64")
    date_died = keras.Input(shape=(1,), name="date_died", dtype="int64")
    intubed = keras.Input(shape=(1,), name="intubed", dtype="int64")
    pneumonia = keras.Input(shape=(1,), name="pneumonia", dtype="int64")
    pregnant = keras.Input(shape=(1,), name="pregnant", dtype="int64")
    diabetes = keras.Input(shape=(1,), name="diabetes", dtype="int64")
    copd = keras.Input(shape=(1,), name="copd", dtype="int64")
    asthma = keras.Input(shape=(1,), name="asthma", dtype="int64")
    inmsupr = keras.Input(shape=(1,), name="inmsupr", dtype="int64")
    hipertension = keras.Input(shape=(1,), name="hipertension", dtype="int64")
    other_disease = keras.Input(shape=(1,), name="other_disease", dtype="int64")
    cardiovascular = keras.Input(shape=(1,), name="cardiovascular", dtype="int64")
    obesity = keras.Input(shape=(1,), name="obesity", dtype="int64")
    renal_chronic = keras.Input(shape=(1,), name="renal_chronic", dtype="int64")
    tobacco = keras.Input(shape=(1,), name="tobacco", dtype="int64")
    clasiffication_final = keras.Input(shape=(1,), name="clasiffication_final", dtype="int64")
    icu = keras.Input(shape=(1,), name="icu", dtype="int64")

    # Numerical features
    age = keras.Input(shape=(1,), name="age")

    all_inputs = [
        usmer,
        medical_unit,
        sex,
        patient_type,
        date_died,
        intubed,
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
        tobacco,
        clasiffication_final,
        icu
    ]

    # Integer categorical features
    usmer_encoded = encode_categorical_feature(sex, "usmer", train_ds, False)
    medical_unit_encoded = encode_categorical_feature(sex, "medical_unit", train_ds, False)
    sex_encoded = encode_categorical_feature(sex, "sex", train_ds, False)
    patient_type_encoded = encode_categorical_feature(sex, "patient_type", train_ds, False)
    date_died_encoded = encode_categorical_feature(sex, "date_died", train_ds, False)
    intubed_encoded = encode_categorical_feature(sex, "intubed", train_ds, False)
    pneumonia_encoded = encode_categorical_feature(sex, "pneumonia", train_ds, False)
    pregnant_encoded = encode_categorical_feature(sex, "pregnant", train_ds, False)
    diabetes_encoded = encode_categorical_feature(sex, "diabetes", train_ds, False)
    copd_encoded = encode_categorical_feature(sex, "copd", train_ds, False)
    asthma_encoded = encode_categorical_feature(sex, "asthma", train_ds, False)
    inmsupr_encoded = encode_categorical_feature(sex, "inmsupr", train_ds, False)
    hipertension_encoded = encode_categorical_feature(sex, "hipertension", train_ds, False)
    other_disease_encoded = encode_categorical_feature(sex, "other_disease", train_ds, False)
    cardiovascular_encoded = encode_categorical_feature(sex, "cardiovascular", train_ds, False)
    obesity_encoded = encode_categorical_feature(sex, "obesity", train_ds, False)
    renal_chronic_encoded = encode_categorical_feature(sex, "renal_chronic", train_ds, False)
    tobacco_encoded = encode_categorical_feature(sex, "tobacco", train_ds, False)
    clasiffication_final_encoded = encode_categorical_feature(sex, "clasiffication_final", train_ds, False)
    icu_encoded = encode_categorical_feature(sex, "icu", train_ds, False)

    # Numerical features
    age_encoded = encode_numerical_feature(age, "age", train_ds)

    all_features = layers.concatenate(
        [
            usmer_encoded,
            medical_unit_encoded,
            sex_encoded,
            patient_type_encoded,
            date_died_encoded,
            intubed_encoded,
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
            clasiffication_final_encoded,
            icu_encoded
        ]
    )
    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])


if __name__ == '__main__':
    main()

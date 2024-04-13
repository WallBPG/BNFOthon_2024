'''
ML model that, given a COVID-19 patient's current
symptom, status, and medical history, will predict whether the patient is at high
risk or not.

Brydon Wall
BNFOthon 2024
'''
import numpy as np
import pandas as pd
import os
import keras

class COVID_Data:
    def __init__(self, in_path: str, processed_path: str = './Processed Covid Data') -> None:
        self.in_path = in_path
        self.proc_path = processed_path
        self._data = None

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
            'SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'DIABETES',
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
        if not self._data:
            if not os.path.isfile(self.proc_path):
                self._data = self.__process_csv(self.proc_path)
            else:
                self.data = pd.read_csv(self.proc_path)
        return self._data

def main():
    covid_data = COVID_Data('./Covid Data.csv')
    print(covid_data.process_csv()['CLASIFFICATION_FINAL'])

covid_columns = [
    'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED',
    'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
    'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
    'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
]

if __name__ == '__main__':
    main()

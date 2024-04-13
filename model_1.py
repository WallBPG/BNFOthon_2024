'''
ML model that, given a COVID-19 patient's current
symptom, status, and medical history, will predict whether the patient is at high
risk or not.

Brydon Wall
BNFOthon 2024
'''
import numpy as np
import pandas as pd
import keras

class COVID_Data:
    def __init__(self, path: str) -> None:
        self.path = path

    def process_csv(self) -> pd.DataFrame:
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

        covid_df = pd.read_csv(self.path)

        # Reformat data to start at 0
        

        #covid_df['']

        return covid_df

def main():
    covid_data = COVID_Data('./Covid Data.csv')
    print(covid_data.process_csv().columns)

covid_columns = [
    'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED',
    'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
    'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
    'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
]

if __name__ == '__main__':
    main()

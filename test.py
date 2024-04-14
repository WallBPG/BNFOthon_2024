covid_columns = [
    'USMER', 'MEDICAL_UNIT', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED',
    'PNEUMONIA', 'AGE', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
    'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
    'RENAL_CHRONIC', 'TOBACCO', 'CLASIFFICATION_FINAL', 'ICU'
]
for c in covid_columns:
    #print(f'{c.lower()} = keras.Input(shape=(1,), name="{c}", dtype="int64")')
    #print(f'{c.lower()}_encoded,')
    print(f'{c.lower()}_encoded = encode_categorical_feature(sex, "{c}", train_ds, False)')
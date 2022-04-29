import joblib
from sklearn.preprocessing import LabelEncoder


def preprocess_data(data):
    data = data[data.marital != 'unknown']
    data = data[data.job != 'unknown']
    data = data[data.housing != 'unknown']
    return data


def encode_data(data):
    le = LabelEncoder()

    categorical_cols = ['month', 'education', 'day_of_week', 'job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']
    data[categorical_cols] = data[categorical_cols].apply(le.fit_transform)

    joblib.dump(le, 'model/featureEncoder.joblib', compress=9)

    return data


def encode_target(y):
    le = LabelEncoder()
    y = le.fit_transform(y)

    joblib.dump(le, 'model/targetEncoder.joblib', compress=9)

    return y

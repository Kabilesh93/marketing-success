from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(data):
    data = data[data.marital != 'unknown']
    data = data[data.job != 'unknown']
    data = data[data.housing != 'unknown']

    le = LabelEncoder()
    enc_edu = OrdinalEncoder(
        categories=['unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course',
                    'university.degree'])
    data['job'] = le.fit_transform(data['job'])
    data['marital'] = le.fit_transform(data['marital'])
    data['default'] = le.fit_transform(data['default'])
    data['housing'] = le.fit_transform(data['housing'])
    data['loan'] = le.fit_transform(data['loan'])
    data['contact'] = le.fit_transform(data['contact'])
    data['month'] = le.fit_transform(data['month'])
    data['day_of_week'] = le.fit_transform(data['day_of_week'])
    data['poutcome'] = le.fit_transform(data['poutcome'])
    data['y'] = le.fit_transform(data['y'])
    data['education'] = le.fit_transform(data['education'])

    return data

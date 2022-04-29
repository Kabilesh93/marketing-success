from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


def preprocess_data(data):
    data = data[data.marital != 'unknown']
    data = data[data.job != 'unknown']
    data = data[data.housing != 'unknown']

    le = LabelEncoder()
    education_mapper = {"unknown": 0, "illiterate": 1, "basic.4y": 3, "basic.6y": 4, "basic.9y": 5, "high.school": 6,
                        "professional.course": 7, "university.degree": 8}
    month_mapper = {"jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5, "jul": 6, "aug": 7, "sep": 8, "oct": 9,
                    "nov": 10, "dec": 11}
    day_mapper = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}
    data['month'] = data['month'].replace(month_mapper)
    data['education'] = data['education'].replace(education_mapper)
    data['day_of_week'] = data['day_of_week'].replace(day_mapper)
    data['job'] = le.fit_transform(data['job'])
    data['marital'] = le.fit_transform(data['marital'])
    data['default'] = le.fit_transform(data['default'])
    data['housing'] = le.fit_transform(data['housing'])
    data['loan'] = le.fit_transform(data['loan'])
    data['contact'] = le.fit_transform(data['contact'])
    data['poutcome'] = le.fit_transform(data['poutcome'])
    data['y'] = le.fit_transform(data['y'])

    return data

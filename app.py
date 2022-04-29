import json
import pickle

import joblib
import pandas as pd
from flask import Flask, request
from flask_restx import Api
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from model_building import *
from preprocess import preprocess_data, encode_data, encode_target

app = Flask(__name__)
api = Api(app)

dataset = pd.read_csv('data/bank-additional-full.csv', sep=';')


@app.route('/marketing-success/train-model', methods=['PUT'])
def train_model():

    data = preprocess_data(dataset)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = encode_data(X).values
    y = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    model = build_random_forest()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Model evaluation results')
    print('Accuracy score : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    print('Precision : {0:0.4f}'.format(precision_score(y_test, y_pred, average="macro")))
    print('Recall : {0:0.4f}'.format(recall_score(y_test, y_pred, average="macro")))
    print('F1-core : {0:0.4f}'.format(f1_score(y_test, y_pred, average="macro")))

    with open('model/trained_model', 'wb') as files:
        pickle.dump(model, files)

    return 'Train model: Completed'


@app.route('/marketing-success/evaluate-models', methods=['PUT'])
def evaluate_models():
    data = preprocess_data(dataset)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = encode_data(X).values
    y = encode_target(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    random_forest_params = [{}]
    logistic_regresion_params = [{}]
    naive_bayes_params = [{}]
    svm_params = [{}]

    model_classes = [
        ["random_forest", build_random_forest, random_forest_params],
        # ["logistic_regresion", build_logistic_regresion, logistic_regresion_params],
        ["naive_bayes", build_naive_bayes, naive_bayes_params]
        # ["svm", build_svm, svm_params]
    ]

    insights = []
    for modelname, Model, params_list in model_classes:
        for params in params_list:
            model = Model(**params)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            insights.append((modelname, model, params, score))

    insights.sort(key=lambda x:x[-1], reverse=True)

    for modelname, model, params, score in insights:
        print(modelname, params, score)

    return "Success"


@app.route('/marketing-success/predict', methods=['GET', 'POST'])
def predict_for_input():
    with open('model/trained_model', 'rb') as f:
        model = pickle.load(f)

    featureEncoder = joblib.load('model/featureEncoder.joblib')
    targetEncoder = joblib.load('model/targetEncoder.joblib')

    categorical_cols = ['month', 'education', 'day_of_week', 'job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']

    content = request.json

    input_data = {
          'age': content['age'],
          'job': content['job'],
          'marital': content['marital'],
          'education': content['education'],
          'default': content['default'],
          'housing': content['housing'],
          'loan': content['loan'],
          'contact': content['contact'],
          'month': content['month'],
          'day_of_week': content['day_of_week'],
          'duration': content['duration'],
          'campaign': content['campaign'],
          'pdays': content['pdays'],
          'previous': content['previous'],
          'poutcome': content['poutcome'],
          'emp.var.rate': content['emp.var.rate'],
          'cons.price.idx': content['cons.price.idx'],
          'cons.conf.idx': content['cons.conf.idx'],
          'euribor3m': content['euribor3m'],
          'nr.employed': content['nr.employed']
    }

    dataframe = pd.DataFrame(input_data, index=[0])

    dataframe[categorical_cols] = dataframe[categorical_cols].apply(featureEncoder.fit_transform)

    prediction = model.predict(dataframe.values)

    predicted_label = targetEncoder.inverse_transform(prediction)

    result = {
        "Predicted Label": predicted_label[0]
    }
    json_result = json.dumps(result, indent=2)

    return json_result


if __name__ == '__main__':
    app.run()

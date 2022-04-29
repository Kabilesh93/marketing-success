from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle

from preprocess import preprocess_data
from model_building import *

app = Flask(__name__)

dataset = pd.read_csv('data/bank-additional-full.csv', sep=';')


@app.route('/marketing-success/train-model')
def train_model():

    data = preprocess_data(dataset)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

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


@app.route('/marketing-success/predict', methods=['GET', 'POST'])
def evaluate_model():
    with open('model/trained_model', 'rb') as f:
        model = pickle.load(f)

    content = request.json
    input_data = {
          'age': [41],
          'job': ['blue-collar'],
          'marital': ['married'],
          'education': ['unknown'],
          'default': ['unknown'],
          'housing': ['no'],
          'loan': ['no'],
          'contact': ['telephone'],
          'month': ['may'],
          'day_of_week': ['mon'],
          'duration': [55],
          'campaign': [1],
          'pdays': [999],
          'previous': [0],
          'poutcome': ['nonexistent'],
          'emp.var.rate': [1.1],
          'cons.price.idx': [93.994],
          'cons.conf.idx': [-36.4],
          'euribor3m': [4.857],
          'nr.employed': [5191]
    }

    df = pd.DataFrame(input_data)

    return "success"


if __name__ == '__main__':
    app.run()

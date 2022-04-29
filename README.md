# Predicting the success of marketing campaign

## Dataset
The dataset consists of direct marketing campaigns data of a banking 
institution, and was picked from UCI Machine Learning Repository. 

Dataset - https://archive.ics.uci.edu/ml/datasets/bank+marketing

## Running the application
- Clone code
- Create a virtual environment inside project directory. 
- Run ```pip install requirements.txt```
- Run ```python app.py```
- The application will start in localhost port 5000

## API Calls

### Train Model
- PUT request to http://127.0.0.1:5000/marketing-success/train-model
- Body - None

### Evaluate ML Algorithms
- PUT request to http://127.0.0.1:5000/marketing-success/evaluate-models
- Body - None

### Make prediction
- POST request to http://127.0.0.1:5000/marketing-success/predict
- Body
```yaml
{
  "age": 41,
  "job": "blue-collar",
  "marital": "married",
  "education": "unknown",
  "default": "unknown",
  "housing": "no",
  "loan": "no",
  "contact": "telephone",
  "month": "may",
  "day_of_week": "mon",
  "duration": 55,
  "campaign": 1,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp.var.rate": 1.1,
  "cons.price.idx": 93.994,
  "cons.conf.idx": -36.4,
  "euribor3m": 4.857,
  "nr.employed": 5191.0
}
```
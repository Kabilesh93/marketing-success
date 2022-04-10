from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# import xgboost as xgb
from sklearn.feature_selection import RFE


def build_random_forest():
    ran_fr = RandomForestClassifier(n_estimators=20, random_state=42)
    rfe_selector = RFE(estimator=ran_fr, n_features_to_select=10, step=4)
    return rfe_selector


def logistic_regresion():
    logReg = LogisticRegression(random_state=42, max_iter=250)
    return logReg


def build_naive_bayes():
    gnb = GaussianNB()
    return gnb


def build_svm():
    svm = SVC(gamma='auto')
    return svm


# def build_xg_boost():
#     xg_class = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
#                                max_depth=5, alpha=10, n_estimators=10)
#     return xg_class

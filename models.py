import os
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
# For loky/backend physical-core detection warning:
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import librosa
from voice_processing import convert_to_wav, extract_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# Classic and ensemble classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Linear & discriminant classifiers
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)

# Probabilistic & instance-based
from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB,
    ComplementNB,
    MultinomialNB
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

# Multi-class wrappers
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Kernel approximation
from sklearn.kernel_approximation import Nystroem, RBFSampler


def train_model(model, X_train, X_test, y_train, y_test, scaler=None):
    """Generic model trainer and multi-metric evaluator."""
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    metrics = {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1_score':  f1
    }
    return model, metrics, scaler


def get_models():
    """Instantiate all classifiers with fixed warnings and regularization."""
    # SVM grid search wrapper
    svm = SVC(kernel='rbf')
    svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm_grid = GridSearchCV(svm, svm_params, cv=3)

    return {
        'Random Forest':                  (RandomForestClassifier(), None),
        'Extra Trees':                    (ExtraTreesClassifier(), None),
        'AdaBoost':                       (AdaBoostClassifier(), None),
        'Gradient Boosting':              (GradientBoostingClassifier(), None),
        'Hist Gradient Boosting':         (HistGradientBoostingClassifier(), None),
        'XGBoost':                        (XGBClassifier(
                                             n_estimators=100,
                                             eval_metric='mlogloss',
                                             verbosity=0
                                         ), None),
        'LightGBM':                       (LGBMClassifier(
                                             verbosity=-1,
                                             force_col_wise=True
                                         ), None),
        'CatBoost':                       (CatBoostClassifier(verbose=0), None),
        'Decision Tree':                  (DecisionTreeClassifier(), None),
        'SGD Classifier':                (SGDClassifier(max_iter=1000, tol=1e-3), StandardScaler()),
        'Passive Aggressive':             (PassiveAggressiveClassifier(max_iter=1000), StandardScaler()),
        'Ridge Classifier':               (RidgeClassifier(), StandardScaler()),
        'Nearest Centroid':               (NearestCentroid(), None),

        'k-NN':                           (KNeighborsClassifier(n_neighbors=5), StandardScaler()),
        'GaussianNB':                     (GaussianNB(), None),
        'BernoulliNB':                    (Pipeline([
                                             ('minmax', MinMaxScaler(feature_range=(0,1))),
                                             ('clf', BernoulliNB())
                                         ]), None),
        'ComplementNB':                   (Pipeline([
                                             ('minmax', MinMaxScaler(feature_range=(0,1))),
                                             ('clf', ComplementNB())
                                         ]), None),
        'MultinomialNB':                  (Pipeline([
                                             ('minmax', MinMaxScaler(feature_range=(0,1))),
                                             ('clf', MultinomialNB())
                                         ]), None),
        'SVM (RBF)':                      (svm_grid, StandardScaler()),
        'Kernel Approx SVM':              (Pipeline([
                                             ('sampler', RBFSampler()),
                                             ('scaler', StandardScaler()),
                                             ('svc', LinearSVC(max_iter=1000))
                                         ]), None),
        'Nystroem SVM':                   (Pipeline([
                                             ('nystroem', Nystroem()),
                                             ('scaler', StandardScaler()),
                                             ('svc', LinearSVC(max_iter=1000))
                                         ]), None),
        'One-vs-Rest LR':                (OneVsRestClassifier(LogisticRegression(max_iter=1000)), StandardScaler()),
        'One-vs-One LR':                 (OneVsOneClassifier(LogisticRegression(max_iter=1000)), StandardScaler()),
        'Linear Discriminant Analysis':   (LinearDiscriminantAnalysis(), None),
        'Quadratic Discriminant Analysis':(QuadraticDiscriminantAnalysis(reg_param=0.1), None),
        'Gaussian Process':               (GaussianProcessClassifier(), None),
        'MLP':                            (MLPClassifier(hidden_layer_sizes=(100,), max_iter=500), StandardScaler())
    }


def predict_audio(file_path, model, scaler=None):
    """Convert, extract features, (optionally) scale, and predict."""
    wav_path = convert_to_wav(file_path)
    if not wav_path:
        return "Invalid audio"
    audio, sr = librosa.load(wav_path, res_type='kaiser-fast')
    feats = extract_features(audio, sr)
    if feats is None:
        return "Could not extract features"
    feats = [feats]
    if scaler:
        feats = scaler.transform(feats)
    pred = model.predict(feats)[0]
    return pred

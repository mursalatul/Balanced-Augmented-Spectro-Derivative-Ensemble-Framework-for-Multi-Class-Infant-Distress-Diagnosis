import os
import warnings

# Suppress specific user warnings to clean up console output
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Limit CPU usage warning fix for parallel backends
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Audio feature extraction tools
import librosa
from voice_processing import convert_to_wav, extract_features

# Evaluation and preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# Import many different classifiers for experimentation
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    PassiveAggressiveClassifier,
    RidgeClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

# Wrappers for handling multi-class classification
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# For kernel approximations (used in custom SVM pipelines)
from sklearn.kernel_approximation import Nystroem, RBFSampler


def train_model(model, X_train, X_test, y_train, y_test, scaler=None):
    """
    Train a given model, apply optional feature scaling, and evaluate it on test data.

    Parameters:
        model: scikit-learn compatible model
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training labels
        y_test (array): Test labels
        scaler (optional): Feature scaler to normalize inputs

    Returns:
        tuple: (trained model, metrics dictionary, used scaler)
    """
    if scaler:
        # Apply feature scaling if a scaler is provided
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit the model on training data
    model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test)

    # Evaluate performance using common classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Collect and return metrics
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    return model, metrics, scaler


def get_models():
    """
    Define and return a dictionary of various machine learning classifiers
    along with their required feature scalers (if any).

    Returns:
        dict: {Model name: (Model instance, Scaler instance or None)}
    """
    # SVM model with hyperparameter tuning using grid search
    svm = SVC(kernel="rbf")
    svm_params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
    svm_grid = GridSearchCV(svm, svm_params, cv=3)

    return {
        # Ensemble models (tree-based)
        "Random Forest": (RandomForestClassifier(), None),
        "Extra Trees": (ExtraTreesClassifier(), None),
        "AdaBoost": (AdaBoostClassifier(), None),
        "Gradient Boosting": (GradientBoostingClassifier(), None),
        "Hist Gradient Boosting": (HistGradientBoostingClassifier(), None),
        # Gradient boosting variants
        "XGBoost": (
            XGBClassifier(n_estimators=100, eval_metric="mlogloss", verbosity=0),
            None,
        ),
        "LightGBM": (LGBMClassifier(verbosity=-1, force_col_wise=True), None),
        "CatBoost": (CatBoostClassifier(verbose=0), None),
        # Classic classifiers
        "Decision Tree": (DecisionTreeClassifier(), None),
        "SGD Classifier": (SGDClassifier(max_iter=1000, tol=1e-3), StandardScaler()),
        "Passive Aggressive": (
            PassiveAggressiveClassifier(max_iter=1000),
            StandardScaler(),
        ),
        "Ridge Classifier": (RidgeClassifier(), StandardScaler()),
        "Nearest Centroid": (NearestCentroid(), None),
        # Distance-based methods
        "k-NN": (KNeighborsClassifier(n_neighbors=5), StandardScaler()),
        # Naive Bayes classifiers (some require input normalization)
        "GaussianNB": (GaussianNB(), None),
        "BernoulliNB": (
            Pipeline(
                [("minmax", MinMaxScaler(feature_range=(0, 1))), ("clf", BernoulliNB())]
            ),
            None,
        ),
        "ComplementNB": (
            Pipeline(
                [
                    ("minmax", MinMaxScaler(feature_range=(0, 1))),
                    ("clf", ComplementNB()),
                ]
            ),
            None,
        ),
        "MultinomialNB": (
            Pipeline(
                [
                    ("minmax", MinMaxScaler(feature_range=(0, 1))),
                    ("clf", MultinomialNB()),
                ]
            ),
            None,
        ),
        # Support Vector Machines
        "SVM (RBF)": (svm_grid, StandardScaler()),
        # Kernel approximation based classifiers
        "Kernel Approx SVM": (
            Pipeline(
                [
                    ("sampler", RBFSampler()),
                    ("scaler", StandardScaler()),
                    ("svc", LinearSVC(max_iter=1000)),
                ]
            ),
            None,
        ),
        "Nystroem SVM": (
            Pipeline(
                [
                    ("nystroem", Nystroem()),
                    ("scaler", StandardScaler()),
                    ("svc", LinearSVC(max_iter=1000)),
                ]
            ),
            None,
        ),
        # Multi-class wrappers
        "One-vs-Rest LR": (
            OneVsRestClassifier(LogisticRegression(max_iter=1000)),
            StandardScaler(),
        ),
        "One-vs-One LR": (
            OneVsOneClassifier(LogisticRegression(max_iter=1000)),
            StandardScaler(),
        ),
        # Discriminant analysis
        "Linear Discriminant Analysis": (LinearDiscriminantAnalysis(), None),
        "Quadratic Discriminant Analysis": (
            QuadraticDiscriminantAnalysis(reg_param=0.1),
            None,
        ),
        # Probabilistic & kernel-based models
        "Gaussian Process": (GaussianProcessClassifier(), None),
        # Neural network model
        "MLP": (
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
            StandardScaler(),
        ),
    }


def predict_audio(file_path, model, scaler=None):
    """
    Predict the class of a new audio file using a trained model.

    Steps:
    - Convert the file to WAV (if needed)
    - Extract features from the audio
    - Apply scaling if a scaler is provided
    - Predict using the given model

    Parameters:
        file_path (str): Path to the input audio file
        model: Trained classifier
        scaler (optional): Scaler used during training

    Returns:
        int or str: Predicted label index or error message
    """
    wav_path = convert_to_wav(file_path)
    if not wav_path:
        return "Invalid audio"

    # Load audio and extract features
    audio, sr = librosa.load(wav_path, res_type="kaiser-fast")
    feats = extract_features(audio, sr)
    if feats is None:
        return "Could not extract features"

    feats = [feats]  # Wrap in list for model input shape
    if scaler:
        feats = scaler.transform(feats)

    # Predict and return the label
    pred = model.predict(feats)[0]
    return pred

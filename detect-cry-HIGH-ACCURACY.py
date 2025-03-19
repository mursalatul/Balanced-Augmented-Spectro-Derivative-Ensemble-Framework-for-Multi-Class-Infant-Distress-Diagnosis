import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pydub import AudioSegment
import warnings
warnings.filterwarnings('ignore')

def convert_to_wav(file_path):
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.wav':
            return file_path
        wav_path = os.path.splitext(file_path)[0] + '.wav'
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print(f"Error converting {file_path}: {str(e)}")
        return None

def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    features = np.hstack([np.mean(mfccs.T, axis=0), np.std(mfccs.T, axis=0), 
                          np.max(mfccs.T, axis=0), np.mean(delta_mfccs.T, axis=0), 
                          np.mean(delta2_mfccs.T, axis=0)])
    return features

def augment_audio(audio, sample_rate):
    audio_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)
    audio_stretch = librosa.effects.time_stretch(audio, rate=1.1)
    audio_noise = audio + 0.005 * np.random.randn(len(audio))
    return [audio_pitch, audio_stretch, audio_noise]

def prepare_dataset(folder_names, base_path="."):
    X = []
    y = []
    for label, folder in enumerate(folder_names):
        folder_path = os.path.join(base_path, folder)
        print(f"\nProcessing folder: {folder_path}")
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            continue
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
        print(f"Found {len(files)} audio files in {folder}")
        for file in files:
            file_path = os.path.join(folder_path, file)
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            features = extract_features(audio, sample_rate)
            if features is not None:
                X.append(features)
                y.append(label)
                for aug_audio in augment_audio(audio, sample_rate):
                    aug_features = extract_features(aug_audio, sample_rate)
                    X.append(aug_features)
                    y.append(label)
    print(f"\nTotal samples processed: {len(X)}")
    return np.array(X), np.array(y)

# Training functions (unchanged except RF with tuning)
def train_random_forest(X_train, X_test, y_train, y_test):
    print("\nTraining Random Forest classifier with tuning...")
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    rf_grid.fit(X_train, y_train)
    y_pred = rf_grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy * 100:.2f}% (Best params: {rf_grid.best_params_})")
    return rf_grid.best_estimator_

def train_xgboost(X_train, X_test, y_train, y_test):
    print("\nTraining XGBoost classifier...")
    xgb_classifier = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")
    return xgb_classifier

def train_svm_rbf(X_train, X_test, y_train, y_test):
    print("\nTraining SVM (RBF Kernel) classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm_classifier = SVC(kernel='rbf', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)
    y_pred = svm_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM (RBF) Accuracy: {accuracy * 100:.2f}%")
    return svm_classifier, scaler

def train_logistic_regression(X_train, X_test, y_train, y_test):
    print("\nTraining Logistic Regression classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
    lr_classifier.fit(X_train_scaled, y_train)
    y_pred = lr_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
    return lr_classifier, scaler

def train_knn(X_train, X_test, y_train, y_test):
    print("\nTraining k-Nearest Neighbors classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_scaled, y_train)
    y_pred = knn_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k-NN Accuracy: {accuracy * 100:.2f}%")
    return knn_classifier, scaler

def train_gradient_boosting(X_train, X_test, y_train, y_test):
    print("\nTraining Gradient Boosting classifier...")
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_classifier.fit(X_train, y_train)
    y_pred = gb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Accuracy: {accuracy * 100:.2f}%")
    return gb_classifier

def train_naive_bayes(X_train, X_test, y_train, y_test):
    print("\nTraining Gaussian Naive Bayes classifier...")
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred = nb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy * 100:.2f}%")
    return nb_classifier

def train_mlp(X_train, X_test, y_train, y_test):
    print("\nTraining Multi-Layer Perceptron classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp_classifier.fit(X_train_scaled, y_train)
    y_pred = mlp_classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"MLP Accuracy: {accuracy * 100:.2f}%")
    return mlp_classifier, scaler

def main():
    folder_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    print("Starting dataset preparation...")
    X, y = prepare_dataset(folder_names)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    print(f"Feature vector size: {X_train.shape[1]}")
    
    accuracies = {}
    rf_classifier = train_random_forest(X_train, X_test, y_train, y_test)
    accuracies['rf'] = accuracy_score(y_test, rf_classifier.predict(X_test))
    xgb_classifier = train_xgboost(X_train, X_test, y_train, y_test)
    accuracies['xgb'] = accuracy_score(y_test, xgb_classifier.predict(X_test))
    svm_classifier, svm_scaler = train_svm_rbf(X_train, X_test, y_train, y_test)
    accuracies['svm'] = accuracy_score(y_test, svm_classifier.predict(svm_scaler.transform(X_test)))
    lr_classifier, lr_scaler = train_logistic_regression(X_train, X_test, y_train, y_test)
    accuracies['lr'] = accuracy_score(y_test, lr_classifier.predict(lr_scaler.transform(X_test)))
    knn_classifier, knn_scaler = train_knn(X_train, X_test, y_train, y_test)
    accuracies['knn'] = accuracy_score(y_test, knn_classifier.predict(knn_scaler.transform(X_test)))
    gb_classifier = train_gradient_boosting(X_train, X_test, y_train, y_test)
    accuracies['gb'] = accuracy_score(y_test, gb_classifier.predict(X_test))
    nb_classifier = train_naive_bayes(X_train, X_test, y_train, y_test)
    accuracies['nb'] = accuracy_score(y_test, nb_classifier.predict(X_test))
    mlp_classifier, mlp_scaler = train_mlp(X_train, X_test, y_train, y_test)
    accuracies['mlp'] = accuracy_score(y_test, mlp_classifier.predict(mlp_scaler.transform(X_test)))
    
    print("\nModel Accuracies (High to Low):")
    for model_name, accuracy in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name.upper()}: {accuracy * 100:.2f}%")
    
    def predict_audio(file_path, classifier, scaler=None):
        audio, sample_rate = librosa.load(convert_to_wav(file_path), res_type='kaiser_fast')
        features = extract_features(audio, sample_rate)
        if features is not None:
            if scaler is not None:
                features = scaler.transform([features])
            else:
                features = [features]
            prediction = classifier.predict(features)[0]
            return folder_names[prediction]
        return None
    
    return {
        'rf': (rf_classifier, None),
        'xgb': (xgb_classifier, None),
        'svm': (svm_classifier, svm_scaler),
        'lr': (lr_classifier, lr_scaler),
        'knn': (knn_classifier, knn_scaler),
        'gb': (gb_classifier, None),
        'nb': (nb_classifier, None),
        'mlp': (mlp_classifier, mlp_scaler)
    }, predict_audio

if __name__ == "__main__":
    classifiers, predict_function = main()
    test_file = "D:\defence\model\donateacry_corpus_cleaned_and_updated_data\discomfort\\1309B82C-F146-46F0-A723-45345AFA6EA8-1430703937-1.0-f-48-dc.wav"
    print("\nPredictions for test file:")
    for model_name, (classifier, scaler) in classifiers.items():
        prediction = predict_function(test_file, classifier, scaler)
        print(f"{model_name.upper()} Prediction: {prediction}")
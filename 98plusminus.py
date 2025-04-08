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
import collections
import random

warnings.filterwarnings('ignore')

# -----------------------------
# Audio Processing Utilities
# -----------------------------

def convert_to_wav(file_path):
    """Convert non-WAV files to WAV using pydub."""
    try:
        if file_path.lower().endswith('.wav'):
            return file_path
        wav_path = os.path.splitext(file_path)[0] + '.wav'
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format='wav')
        return wav_path
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

def extract_features(audio, sample_rate):
    """Extract MFCC-based features along with additional spectral features."""
    try:
        # MFCCs and their derivatives
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Additional spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        
        features = np.hstack([
            np.mean(mfccs.T, axis=0),
            np.std(mfccs.T, axis=0),
            np.max(mfccs.T, axis=0),
            np.mean(delta.T, axis=0),
            np.mean(delta2.T, axis=0),
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate
        ])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def augment_audio(audio, sr):
    """Return list of augmented audio versions."""
    return [
        librosa.effects.pitch_shift(audio, sr=sr, n_steps=random.uniform(-2, 2)),
        librosa.effects.time_stretch(audio, rate=random.uniform(0.9, 1.1)),
        audio + 0.005 * np.random.randn(len(audio))
    ]

# -----------------------------
# Dataset Preparation
# -----------------------------

def balance_dataset_with_augmentation(folder_names, base_path="."):
    """
    Load and balance dataset by augmenting minority classes
    to match the size of the majority class.
    """
    class_counts = {}
    class_files = {}

    for label, folder in enumerate(folder_names):
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            continue
        files = [f for f in os.listdir(path) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
        class_counts[label] = len(files)
        class_files[label] = files

    max_class = max(class_counts, key=class_counts.get)
    max_samples = class_counts[max_class]
    print(f"\n📌 Max class: {folder_names[max_class]} with {max_samples} samples.")

    X, y = [], []

    for label, files in class_files.items():
        folder_path = os.path.join(base_path, folder_names[label])
        current_features = []

        for file in files:
            file_path = os.path.join(folder_path, file)
            wav_path = convert_to_wav(file_path)
            if not wav_path:
                continue
            try:
                audio, sr = librosa.load(wav_path, res_type='kaiser_fast')
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                continue
            features = extract_features(audio, sr)
            if features is not None:
                current_features.append(features)

        X.extend(current_features)
        y.extend([label] * len(current_features))

        # Augment data for minority classes
        if label == max_class:
            continue

        current_count = len(current_features)
        while current_count < max_samples:
            for file in files:
                file_path = os.path.join(folder_path, file)
                wav_path = convert_to_wav(file_path)
                if not wav_path:
                    continue
                try:
                    audio, sr = librosa.load(wav_path, res_type='kaiser_fast')
                except Exception as e:
                    print(f"Error loading {wav_path}: {e}")
                    continue
                for aug_audio in augment_audio(audio, sr):
                    aug_features = extract_features(aug_audio, sr)
                    if aug_features is not None:
                        X.append(aug_features)
                        y.append(label)
                        current_count += 1
                    if current_count >= max_samples:
                        break
                if current_count >= max_samples:
                    break

    return np.array(X), np.array(y)

# -----------------------------
# Model Training
# -----------------------------

def train_model(model, X_train, X_test, y_train, y_test, scaler=None):
    """Generic model trainer and evaluator."""
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, scaler

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    # No fixed seeds here to allow variability in each run
    folder_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    print("📦 Preparing balanced dataset with augmentation...")
    X, y = balance_dataset_with_augmentation(folder_names)

    # Show class distribution
    label_counts = collections.Counter(y)
    print("\n📊 Class distribution after augmentation:")
    for label, count in sorted(label_counts.items()):
        print(f"{folder_names[label]}: {count}")

    # Split the dataset without a fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    print(f"\n🔹 Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"🔹 Feature size: {X.shape[1]}")

    # SVM is tuned using GridSearchCV with a small parameter grid.
    svm_model = SVC(kernel='rbf')
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm_grid = GridSearchCV(svm_model, param_grid, cv=3)
    
    models = {
        'Random Forest': (RandomForestClassifier(), None),
        'XGBoost': (XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss'), None),
        'SVM (RBF)': (svm_grid, StandardScaler()),
        'Logistic Regression': (LogisticRegression(max_iter=1000), StandardScaler()),
        'k-NN': (KNeighborsClassifier(n_neighbors=5), StandardScaler()),
        'Gradient Boosting': (GradientBoostingClassifier(), None),
        'Naive Bayes': (GaussianNB(), None),
        'MLP': (MLPClassifier(hidden_layer_sizes=(100,), max_iter=500), StandardScaler())
    }

    results = {}
    trained_models = {}

    for name, (model, scaler) in models.items():
        print(f"\n🚀 Training {name}...")
        trained_model, acc, trained_scaler = train_model(model, X_train, X_test, y_train, y_test, scaler)
        results[name] = acc
        trained_models[name] = (trained_model, trained_scaler)
        print(f"{name} Accuracy: {acc * 100:.2f}%")

    print("\n📈 Final Model Accuracies (sorted):")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc * 100:.2f}%")

    def predict_audio(file_path, model, scaler=None):
        wav_path = convert_to_wav(file_path)
        if not wav_path:
            return "Invalid audio"
        audio, sr = librosa.load(wav_path, res_type='kaiser_fast')
        features = extract_features(audio, sr)
        if features is None:
            return "Could not extract features"
        features = [features]
        if scaler:
            features = scaler.transform(features)
        pred = model.predict(features)[0]
        return folder_names[pred]

    return results, trained_models, predict_audio

# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    best_accuracies = []
    runtimes = 1  # Adjust the number of runs as needed
    for i in range(runtimes):
        results, trained_models, predict_func = main()
        best_accuracy = max(results.values())
        best_accuracies.append(best_accuracy)
    print("\nBest Accuracies over runs:")
    for idx, acc in enumerate(best_accuracies, 1):
        print(f"Run {idx}: {acc * 100:.2f}%")

import os
import numpy as np
import librosa
from pydub import AudioSegment
import random

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
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        spectral_rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
        zero_crossing_rate= np.mean(librosa.feature.zero_crossing_rate(y=audio))

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

def balance_dataset_with_augmentation(folder_names, base_path="."):
    """
    Load and balance dataset by augmenting minority classes
    to match the size of the majority class.
    """
    class_counts = {}
    class_files  = {}

    for label, folder in enumerate(folder_names):
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            continue
        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.wav', '.mp3', '.m4a', '.ogg'))]
        class_counts[label] = len(files)
        class_files[label] = files

    max_class   = max(class_counts, key=class_counts.get)
    max_samples = class_counts[max_class]
    print(f"\n📌 Max class: {folder_names[max_class]} with {max_samples} samples.")

    X, y = [], []
    for label, files in class_files.items():
        folder_path = os.path.join(base_path, folder_names[label])
        current_features = []

        # extract features for original files
        for file in files:
            file_path = os.path.join(folder_path, file)
            wav_path  = convert_to_wav(file_path)
            if not wav_path:
                continue
            try:
                audio, sr = librosa.load(wav_path, res_type='kaiser_fast')
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                continue
            feats = extract_features(audio, sr)
            if feats is not None:
                current_features.append(feats)

        X.extend(current_features)
        y.extend([label] * len(current_features))

        # augment minority classes
        if label == max_class:
            continue

        current_count = len(current_features)
        while current_count < max_samples:
            for file in files:
                file_path = os.path.join(folder_path, file)
                wav_path  = convert_to_wav(file_path)
                if not wav_path:
                    continue
                try:
                    audio, sr = librosa.load(wav_path, res_type='kaiser_fast')
                except Exception as e:
                    print(f"Error loading {wav_path}: {e}")
                    continue
                for aug_audio in augment_audio(audio, sr):
                    aug_feats = extract_features(aug_audio, sr)
                    if aug_feats is not None:
                        X.append(aug_feats)
                        y.append(label)
                        current_count += 1
                    if current_count >= max_samples:
                        break
                if current_count >= max_samples:
                    break

    return np.array(X), np.array(y)

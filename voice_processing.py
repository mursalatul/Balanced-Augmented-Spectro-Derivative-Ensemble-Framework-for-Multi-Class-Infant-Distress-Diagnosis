import os  # Provides a way of using operating system-dependent functionality like file paths
import numpy as np  # Used for numerical operations and handling arrays
import librosa  # Used for audio processing
from pydub import AudioSegment  # Used for audio file conversion
import random  # Used to generate random numbers for augmentation


def convert_to_wav(file_path):
    """
    Converts an audio file to WAV format if it's not already in WAV.

    Parameters:
        file_path (str): Path to the input audio file.

    Returns:
        str or None: Path to the WAV file, or None if conversion fails.
    """
    try:
        # If already a WAV file, just return it
        if file_path.lower().endswith(".wav"):
            return file_path

        # wav_path (str): New file path with .wav extension
        wav_path = os.path.splitext(file_path)[0] + ".wav"

        # audio (AudioSegment): Loaded audio using pydub
        audio = AudioSegment.from_file(file_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None


def extract_features(audio, sample_rate):
    """
    Extracts audio features to help train a machine learning model.

    Parameters:
        audio (np.array): The audio time series.
        sample_rate (int): The sample rate of the audio.

    Returns:
        np.array or None: Extracted features as a 1D array or None on error.
    """
    try:
        # mfccs (np.ndarray): 40 Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # delta (np.ndarray): First-order derivative of MFCCs
        delta = librosa.feature.delta(mfccs)

        # delta2 (np.ndarray): Second-order derivative of MFCCs
        delta2 = librosa.feature.delta(mfccs, order=2)

        # spectral_centroid (float): Mean of spectral centroid values
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        )

        # spectral_rolloff (float): Mean of spectral rolloff values
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        )

        # zero_crossing_rate (float): Mean of zero crossing rate values
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # features (np.ndarray): Combined feature vector
        features = np.hstack(
            [
                np.mean(mfccs.T, axis=0),  # Average of MFCCs
                np.std(mfccs.T, axis=0),  # Standard deviation of MFCCs
                np.max(mfccs.T, axis=0),  # Maximum value of MFCCs
                np.mean(delta.T, axis=0),  # Average of 1st derivative
                np.mean(delta2.T, axis=0),  # Average of 2nd derivative
                spectral_centroid,
                spectral_rolloff,
                zero_crossing_rate,
            ]
        )
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def augment_audio(audio, sr):
    """
    Generates augmented versions of an audio file to increase data diversity.

    Parameters:
        audio (np.array): Original audio time series.
        sr (int): Sample rate of the audio.

    Returns:
        list: List of augmented audio arrays.
    """
    return [
        librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=random.uniform(-2, 2)
        ),  # Shift pitch randomly
        librosa.effects.time_stretch(
            audio, rate=random.uniform(0.9, 1.1)
        ),  # Stretch or compress audio slightly
        audio + 0.005 * np.random.randn(len(audio)),  # Add random noise
    ]


def balance_dataset_with_augmentation(folder_names, base_path="."):
    """
    Loads audio files from folders, extracts features, and balances the dataset by
    augmenting audio from under-represented classes until all classes have equal samples.

    Parameters:
        folder_names (list of str): Names of folders, each representing a class.
        base_path (str): Base directory containing the folders.

    Returns:
        tuple: (X, y)
            X (np.array): Feature matrix.
            y (np.array): Corresponding class labels.
    """
    # class_counts (dict): Stores the number of files in each class
    class_counts = {}

    # class_files (dict): Stores the list of file names per class
    class_files = {}

    # label (int): Integer label for the class
    # folder (str): Folder name corresponding to a class
    for label, folder in enumerate(folder_names):
        # path (str): Full path to the folder
        path = os.path.join(base_path, folder)

        if not os.path.exists(path):
            continue

        # files (list): List of supported audio file names
        files = [
            f
            for f in os.listdir(path)
            if f.lower().endswith((".wav", ".mp3", ".m4a", ".ogg"))
        ]

        class_counts[label] = len(files)
        class_files[label] = files

    # max_class (int): Label of the class with maximum samples
    max_class = max(class_counts, key=class_counts.get)

    # max_samples (int): Number of samples in the majority class
    max_samples = class_counts[max_class]

    print(f"\n📌 Max class: {folder_names[max_class]} with {max_samples} samples.")

    # X (list): List to hold feature vectors
    # y (list): List to hold corresponding labels
    X, y = [], []

    for label, files in class_files.items():
        # folder_path (str): Full path to the folder of current class
        folder_path = os.path.join(base_path, folder_names[label])

        # current_features (list): Holds extracted features for the current class
        current_features = []

        # file (str): Name of the audio file
        for file in files:
            # file_path (str): Full path to the audio file
            file_path = os.path.join(folder_path, file)

            # wav_path (str or None): Converted WAV file path
            wav_path = convert_to_wav(file_path)
            if not wav_path:
                continue

            try:
                # audio (np.ndarray 1D): Audio waveform signal
                # sr (int): Sampling rate
                audio, sr = librosa.load(wav_path, res_type="kaiser_fast")
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                continue

            # feats (np.array): Extracted features
            feats = extract_features(audio, sr)
            if feats is not None:
                current_features.append(feats)

        # Add features and labels to final lists
        X.extend(current_features)
        y.extend([label] * len(current_features))

        # current_count (int): Number of samples currently extracted for this class
        current_count = len(current_features)

        # Skip augmentation for the majority class
        if label == max_class:
            continue

        # Augment until class is balanced
        while current_count < max_samples:
            for file in files:
                file_path = os.path.join(folder_path, file)
                wav_path = convert_to_wav(file_path)
                if not wav_path:
                    continue

                try:
                    audio, sr = librosa.load(wav_path, res_type="kaiser_fast")
                except Exception as e:
                    print(f"Error loading {wav_path}: {e}")
                    continue

                # aug_audio (np.ndarray): Augmented version of the original audio
                for aug_audio in augment_audio(audio, sr):
                    # aug_feats (np.array): Features from augmented audio
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

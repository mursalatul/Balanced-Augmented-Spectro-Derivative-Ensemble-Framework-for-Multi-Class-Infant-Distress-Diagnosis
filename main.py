import collections
import pickle
from sklearn.model_selection import train_test_split

# Importing from local modules
from voice_processing import balance_dataset_with_augmentation
from models import get_models, train_model, predict_audio


def main():
    """
    Main function to prepare the dataset, train multiple models, evaluate them,
    and save the best-performing model to disk.

    Returns:
        results (dict): Performance metrics for all models.
        trained_models (dict): Trained models and their associated scalers.
        predict_audio (function): Function to predict new audio samples.
    """

    # List of class folders. Each folder contains audio files for one label.
    folder_names = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
    print("📦 Preparing balanced dataset with augmentation...")

    # Load and augment the dataset to make all classes equally represented
    X, y = balance_dataset_with_augmentation(folder_names)

    # Count how many samples each class has after balancing
    label_counts = collections.Counter(y)
    print("\n📊 Class distribution after augmentation:")
    for label, count in sorted(label_counts.items()):
        print(f"{folder_names[label]}: {count}")

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    print(f"\n🔹 Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"🔹 Feature size: {X.shape[1]}")

    # Load the models to train (e.g., SVM, Random Forest, etc.)
    models = get_models()
    results = {}  # Store evaluation metrics for each model
    trained_models = {}  # Store trained models and their scalers

    # Train each model and evaluate its performance
    for name, (model, scaler) in models.items():
        print(f"\n🚀 Training {name}...")
        trained_model, metrics, trained_scaler = train_model(
            model, X_train, X_test, y_train, y_test, scaler
        )
        results[name] = metrics
        trained_models[name] = (trained_model, trained_scaler)

        # Show model metrics
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']*100:.2f}%  "
            f"Precision={metrics['precision']*100:.2f}%  "
            f"Recall={metrics['recall']*100:.2f}%  "
            f"F1={metrics['f1_score']*100:.2f}%"
        )

    # Show all models ranked by accuracy
    print("\n📈 Final Model Metrics (sorted by accuracy):")
    for name, m in sorted(
        results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    ):
        print(
            f"{name}: Acc={m['accuracy']*100:.2f}%  "
            f"Prec={m['precision']*100:.2f}%  "
            f"Rec={m['recall']*100:.2f}%  "
            f"F1={m['f1_score']*100:.2f}%"
        )

    # Identify the best-performing model based on accuracy
    best_model_name, _ = max(results.items(), key=lambda x: x[1]["accuracy"])
    best_model, best_scaler = trained_models[best_model_name]

    # Save the best model and its scaler for future use
    pkl_filename = f"{best_model_name.lower().replace(' ', '_')}_model.pkl"
    with open(pkl_filename, "wb") as f:
        pickle.dump({"model": best_model, "scaler": best_scaler}, f)
    print(f"\n💾 Best model ({best_model_name}) and scaler saved to '{pkl_filename}'")

    # Return all useful components for later use
    return results, trained_models, predict_audio


# This ensures the code runs only when executed directly, not when imported
if __name__ == "__main__":
    results, trained_models, predict_func = main()

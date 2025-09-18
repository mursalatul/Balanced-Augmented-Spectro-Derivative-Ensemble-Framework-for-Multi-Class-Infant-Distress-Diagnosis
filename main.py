import collections
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Importing from local modules
from voice_processing import balance_dataset_with_augmentation
from models import get_models, train_model, predict_audio

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)


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
    # X: Feature matrix (2D numpy array), each row is a feature vector for one sample
    # y: Target labels corresponding to each feature vector
    X, y = balance_dataset_with_augmentation(folder_names)

    # Count how many samples each class has after balancing
    label_counts = collections.Counter(y)
    print("\n📊 Class distribution after augmentation:")
    for label, count in sorted(label_counts.items()):
        print(f"{folder_names[label]}: {count}")

    # Split the dataset into training and testing sets (80% train, 20% test)
    # stratify=y ensures the class distribution is preserved in both splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    print(f"\n🔹 Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"🔹 Feature size: {X.shape[1]}")

    # Load the models to train (e.g., SVM, Random Forest, etc.)
    models = get_models()  # Returns a dictionary: model_name -> (model_object, scaler)
    results = {}  # Store evaluation metrics for each model
    trained_models = {}  # Store trained models and their scalers

    # Train each model and evaluate its performance
    for name, (model, scaler) in models.items():
        print(f"\n🚀 Training {name}...")
        # Perform k-fold cross validation first
        if scaler:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        print(f"\n🔄 5-Fold Cross Validation Results for {name}:")
        print("=" * 50)
        print("Individual Fold Performance:")
        for fold_idx, score in enumerate(cv_scores, 1):
            print(f"  Fold {fold_idx}: {score*100:.2f}% accuracy")
        print("-" * 50)
        print(f"Overall Performance:")
        print(f"  Mean Accuracy: {cv_scores.mean()*100:.2f}%")
        print(f"  Standard Deviation: ±{cv_scores.std()*100:.2f}%")
        print(f"  95% Confidence Interval: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*2*100:.2f}%")
        print("=" * 50)

        # Train the model
        trained_model, metrics, trained_scaler = train_model(
            model, X_train, X_test, y_train, y_test, scaler
        )
        results[name] = metrics
        trained_models[name] = (trained_model, trained_scaler)

        # Generate and save confusion matrix
        if scaler:
            X_test_scaled = trained_scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        y_pred = trained_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {name}')
        plt.colorbar()
        classes = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()

        # Store ROC curve data
        y_test_bin = label_binarize(y_test, classes=np.unique(y))
        if hasattr(trained_model, "predict_proba"):
            y_score = trained_model.predict_proba(X_test_scaled)
        elif hasattr(trained_model, "decision_function"):
            y_score = trained_model.decision_function(X_test_scaled)
        else:
            y_score = label_binarize(trained_model.predict(X_test_scaled), classes=np.unique(y))
        
        results[name]['roc_data'] = {'y_test': y_test_bin, 'y_score': y_score}

        # Show model metrics
        print(
            f"\n📊 Test Set Metrics for {name}:"
            f"\nAccuracy={metrics['accuracy']*100:.2f}%  "
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

    # Generate combined ROC curve
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, metrics), color in zip(results.items(), colors):
        y_test_bin = metrics['roc_data']['y_test']
        y_score = metrics['roc_data']['y_score']
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_test_bin.shape[1]
        
        # Calculate micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label=f'{name} (AUC = {roc_auc["micro"]:.2f})',
            color=color,
            lw=2,
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.savefig('results/combined_roc_curves.png')
    plt.close()

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

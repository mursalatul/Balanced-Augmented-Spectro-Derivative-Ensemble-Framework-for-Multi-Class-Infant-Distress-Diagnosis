# import collections
# from sklearn.model_selection import train_test_split

# from voice_processing import balance_dataset_with_augmentation
# from models import get_models, train_model, predict_audio


# def main():
#     folder_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
#     print("📦 Preparing balanced dataset with augmentation...")
#     X, y = balance_dataset_with_augmentation(folder_names)

#     # Show class distribution
#     label_counts = collections.Counter(y)
#     print("\n📊 Class distribution after augmentation:")
#     for label, count in sorted(label_counts.items()):
#         print(f"{folder_names[label]}: {count}")

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y
#     )
#     print(f"\n🔹 Train size: {len(X_train)}, Test size: {len(X_test)}")
#     print(f"🔹 Feature size: {X.shape[1]}")

#     # Train models and collect metrics
#     models = get_models()
#     results = {}
#     trained_models = {}

#     for name, (model, scaler) in models.items():
#         print(f"\n🚀 Training {name}...")
#         trained_model, metrics, trained_scaler = train_model(
#             model, X_train, X_test, y_train, y_test, scaler
#         )
#         results[name] = metrics
#         trained_models[name] = (trained_model, trained_scaler)
#         print(
#             f"{name}: "
#             f"Accuracy={metrics['accuracy']*100:.2f}%  "
#             f"Precision={metrics['precision']*100:.2f}%  "
#             f"Recall={metrics['recall']*100:.2f}%  "
#             f"F1={metrics['f1_score']*100:.2f}%"
#         )

#     # Summary
#     print("\n📈 Final Model Metrics (sorted by accuracy):")
#     for name, m in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
#         print(
#             f"{name}: Acc={m['accuracy']*100:.2f}%  "
#             f"Prec={m['precision']*100:.2f}%  "
#             f"Rec={m['recall']*100:.2f}%  "
#             f"F1={m['f1_score']*100:.2f}%"
#         )

#     return results, trained_models, predict_audio


# if __name__ == "__main__":
#     results, trained_models, predict_func = main()

import collections
import pandas as pd
from sklearn.model_selection import train_test_split

from voice_processing import balance_dataset_with_augmentation
from models import get_models, train_model, predict_audio


def train_and_evaluate(X, y):
    """
    Perform train/test split, train models, and return metrics dict and trained_models.
    """
    # Show class distribution
    label_counts = collections.Counter(y)
    print("\n📊 Class distribution after augmentation:")
    folder_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    for label, count in sorted(label_counts.items()):
        print(f"{folder_names[label]}: {count}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    print(f"\n🔹 Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"🔹 Feature size: {X.shape[1]}")

    # Train models and collect metrics
    models = get_models()
    results = {}
    trained_models = {}

    for name, (model, scaler) in models.items():
        print(f"\n🚀 Training {name}...")
        trained_model, metrics, trained_scaler = train_model(
            model, X_train, X_test, y_train, y_test, scaler
        )
        results[name] = metrics
        trained_models[name] = (trained_model, trained_scaler)
        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']*100:.2f}%  "
            f"Precision={metrics['precision']*100:.2f}%  "
            f"Recall={metrics['recall']*100:.2f}%  "
            f"F1={metrics['f1_score']*100:.2f}%"
        )

    return results, trained_models


if __name__ == "__main__":
    folder_names = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    print("📦 Preparing balanced dataset with augmentation...")
    X, y = balance_dataset_with_augmentation(folder_names)

    n_runs = 10
    records = []

    for run in range(1, n_runs + 1):
        print(f"=== Run {run}/{n_runs} ===")
        results, trained_models = train_and_evaluate(X, y)
        row = {'run': run}
        for name, metrics in results.items():
            row[f"{name}_acc"] = metrics['accuracy']
            row[f"{name}_prec"] = metrics['precision']
            row[f"{name}_rec"] = metrics['recall']
            row[f"{name}_f1"] = metrics['f1_score']
        records.append(row)

    df = pd.DataFrame(records)
    excel_path = 'model_metrics.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"\nAll runs complete. Metrics saved to {excel_path}.")

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
from auc_roc_eval import evaluate_roc_for_filters as evaluate_roc_auc, plot_roc_curves as plot_roc_curve

try:
    from tuning import load_data_from_split_files
except ImportError:
    print("Warning: load_data_from_split_files not found in tuning.py. Ensure the file exists.")

def load_cached_data(model_type):
    X, y, file_names = [], [], []
    label_map = {"lastfm": 0, "suno": 1, "udio": 1}
    base_folder = "embeddings"

    for d_name, label in label_map.items():
        folder = os.path.join(base_folder, model_type, d_name)
        if not os.path.exists(folder):
            print(f"Skipping {folder}, path not found.")
            continue

        print(f"Loading {d_name} from {folder}...")
        for npy_file in Path(folder).glob("*.npy"):
            emb = np.load(npy_file)
            if np.isnan(emb).any(): 
                continue
            X.append(emb)
            y.append(label)
            file_names.append(f"{d_name}/audio/{npy_file.stem}.mp3")

    return np.array(X), np.array(y), np.array(file_names)

def save_splits_to_txt(train_files, val_files, test_files, feature_name):
    """Saves Train, Val, AND Test file paths."""
    clean_name = feature_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    for name, data in [("train", train_files), ("val", val_files), ("test", test_files)]:
        with open(f"{name}_{clean_name}.txt", "w") as f:
            for item in data:
                f.write(f"{item}\n")
    print(f"Saved 70/10/20 splits for {feature_name}")

def get_train_val_test_splits(X, y, file_names, feature_name):
    # Split off 20% Test
    X_temp, X_test, y_temp, y_test, temp_files, test_files = train_test_split(
        X, y, file_names, test_size=0.2, random_state=42, stratify=y
    )
    # Split 80% into 70% Train / 10% Val
    X_train, X_val, y_train, y_val, train_files, val_files = train_test_split(
        X_temp, y_temp, temp_files, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    save_splits_to_txt(train_files, val_files, test_files, feature_name)
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_classifiers(X, y, file_names, feature_name="Feature"):
    # Try to load from files first
    try:
        X_train, y_train, _ = load_data_from_split_files(feature_name, "train")
        X_test, y_test, _ = load_data_from_split_files(feature_name, "test")
        print(f"Successfully loaded {feature_name} splits from text files.")
    except:
        X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits(X, y, file_names, feature_name)
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifiers = {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma='scale', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7, metric='cosine'),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(256, 128), alpha=0.01, max_iter=500, random_state=42)
    }

    print("\n" + "="*90)
    print(f"Results for {feature_name}")
    print(f"{'Classifier':<20} | {'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Support':<10}")
    print("-" * 90)

    for name, clf in classifiers.items():
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        elapsed = time.time() - start_time
        
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)

        print(f"{name:<20} | {'Non-AI':<10} | {precision[0]:<10.3f} | {recall[0]:<10.3f} | {f1[0]:<10.3f} | {support[0]:<10}")
        print(f"{'(' + f'{elapsed:.2f}s' + ')':<20} | {'AI':<10} | {precision[1]:<10.3f} | {recall[1]:<10.3f} | {f1[1]:<10.3f} | {support[1]:<10}")
        print(f"Confusion Matrix: [TN: {cm[0][0]}, FP: {cm[0][1]}] | [FN: {cm[1][0]}, TP: {cm[1][1]}]")
        print("-" * 90)

        model_filename = f"baseline_{feature_name}_{name}.joblib".replace(" ", "_").lower().replace("-", "_")
        joblib.dump(clf, model_filename)
        joblib.dump(scaler, f"scaler_{feature_name}.joblib".replace(" ", "_").lower().replace("-", "_"))

if __name__ == "__main__":
    for model in ["clap-laion-music", "mert-v0"]:
        X, y, names = load_cached_data(model)
        if len(X) > 0:
            train_classifiers(X, y, names, feature_name=model)

        
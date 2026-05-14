import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler 
from training import load_cached_data
from pathlib import Path
import os 

def load_data_from_split_files(model_type, split_name):
    """
    Loads embeddings for a specific split (train, val, or test) 
    using the filenames stored in the corresponding .txt file.
    """
    X, y, file_names = [], [], []
    label_map = {"lastfm": 0, "suno": 1, "udio": 1}
    base_folder = "embeddings"
    
    clean_name = model_type.lower().replace(" ", "_").replace("(", "").replace(")", "")
    split_file = f"{split_name}.txt" 
    
    if not os.path.exists(split_file):
        split_file = f"{split_name}.txt"
        if not os.path.exists(split_file):
            print(f"Error: Split file {split_file} not found.")
            return np.array([]), np.array([]), []

    with open(split_file, "r") as f:
        target_files = [line.strip() for line in f.readlines()]

    print(f"Loading {len(target_files)} samples for {split_name} ({model_type})...")

    for file_path in target_files:
        parts = file_path.split('/')
        dataset_name = parts[0]
        filename = parts[-1].replace(".mp3", ".npy")
        npy_path = os.path.join(base_folder, model_type, dataset_name, filename)

        if os.path.exists(npy_path):
            emb = np.load(npy_path)
            if not np.isnan(emb).any():
                X.append(emb)
                y.append(label_map[dataset_name])
                file_names.append(file_path)

    return np.array(X), np.array(y), file_names


def tune_hyperparameters(X_train, y_train, X_val, y_val, feature_name):
    """Tunes hyperparameters and returns best models, scores, and their configs."""
    print(f"\nTuning Hyperparameters for {feature_name}")
    best_models = {}

    # SVM Tuning
    print("Tuning SVM...")
    best_svm, best_svm_f1, best_svm_params = None, 0, {}
    for kernel in ['linear', 'rbf']:
        for C in [0.1, 1, 10]:
            for gamma in ['scale', 'auto']:
                clf = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                clf.fit(X_train, y_train)
                val_f1 = f1_score(y_val, clf.predict(X_val))
                if val_f1 > best_svm_f1:
                    best_svm_f1 = val_f1
                    best_svm = clf
                    best_svm_params = {'kernel': kernel, 'C': C, 'gamma': gamma}
    best_models["SVM"] = {"model": best_svm, "val_f1": best_svm_f1, "params": best_svm_params}

    # Random Forest Tuning
    print("Tuning Random Forest...")
    best_rf, best_rf_f1, best_rf_params = None, 0, {}
    for n_est in [100, 300, 500]:
        for depth in [None, 10, 20]:
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, class_weight='balanced', random_state=42)
            clf.fit(X_train, y_train)
            val_f1 = f1_score(y_val, clf.predict(X_val))
            if val_f1 > best_rf_f1:
                best_rf_f1 = val_f1
                best_rf = clf
                best_rf_params = {'n_estimators': n_est, 'max_depth': depth}
    best_models["Random Forest"] = {"model": best_rf, "val_f1": best_rf_f1, "params": best_rf_params}

    # MLP Tuning
    print("Tuning MLP...")
    best_mlp, best_mlp_f1, best_mlp_params = None, 0, {}
    for arch in [(256, 128), (512, 256)]:
        for alpha in [0.0001, 0.001, 0.01, 0.1]:
            clf = MLPClassifier(hidden_layer_sizes=arch, alpha=alpha, max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
            val_f1 = f1_score(y_val, clf.predict(X_val))
            if val_f1 > best_mlp_f1:
                best_mlp_f1 = val_f1
                best_mlp = clf
                best_mlp_params = {'layers': arch, 'alpha': alpha}
    best_models["MLP"] = {"model": best_mlp, "val_f1": best_mlp_f1, "params": best_mlp_params}

    # kNN Tuning
    print("Tuning kNN...")
    best_knn, best_knn_f1, best_knn_params = None, 0, {}
    for k in [3, 5, 7, 11, 21]:
        for metric in ['euclidean', 'manhattan', 'cosine']:
            clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
            clf.fit(X_train, y_train)
            val_f1 = f1_score(y_val, clf.predict(X_val))
            if val_f1 > best_knn_f1:
                best_knn_f1 = val_f1
                best_knn = clf
                best_knn_params = {'k': k, 'metric': metric}
    best_models["kNN"] = {"model": best_knn, "val_f1": best_knn_f1, "params": best_knn_params}


    return best_models


if __name__ == "__main__":
    feature_name = "clap-laion-music"
    
    # Load splits from text files generated during training
    X_train, y_train, _ = load_data_from_split_files(feature_name, "train")
    X_val, y_val, _ = load_data_from_split_files(feature_name, "val")
    
    if len(X_train) > 0 and len(X_val) > 0:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        print(f"Starting hyperparameter tuning for {feature_name}...")
        tuning_results = tune_hyperparameters(X_train, y_train, X_val, y_val, feature_name)
    else:
        print("Error: Could not load training or validation data.")
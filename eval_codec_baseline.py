import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

def load_all_codec_data(model_type, codec_type):
    """
    Loads EVERY embedding from the codec folder 
    4000 AI (Suno/Udio) + 2000 Human (LastFM)
    """
    X, y = [], []
    label_map = {"lastfm": 0, "suno": 1, "udio": 1}
    base_folder = Path("embeddings_codec") / model_type / codec_type

    for d_name, label in label_map.items():
        folder = base_folder / d_name
        if not folder.exists(): 
            continue

        for npy_file in folder.rglob("*.npy"):
            try:
                emb = np.load(npy_file)
                if np.isnan(emb).any(): continue
                X.append(emb)
                y.append(label)
            except Exception:
                continue

    return np.array(X), np.array(y)

def run_full_codec_evaluation(model_label, codec_type, classifier_type):
    """
    Evaluates specific pre-trained classifiers on the FULL codec dataset.
    """
    X_test, y_test = load_all_codec_data(model_label, codec_type)
    if len(X_test) == 0: return None

    clean_label = model_label.replace("-", "_")
    scaler_path = f"scaler_{clean_label}.joblib"
    
    # Mapping classifier types to the exact filenames 
    if classifier_type.lower() == "svm":
        model_path = f"baseline_{clean_label}_svm_(rbf).joblib"
    elif classifier_type.lower() == "mlp":
        model_path = f"baseline_{clean_label}_mlp_(neural_net).joblib"
    elif classifier_type.lower() == "random forest":
        model_path = f"baseline_{clean_label}_random_forest.joblib"
    elif classifier_type.lower() == "knn":
        model_path = f"baseline_{clean_label}_knn.joblib"
    else:
        return None

    try:
        scaler = joblib.load(scaler_path)
        clf = joblib.load(model_path)
    except FileNotFoundError:
        # Fallback: check if the file exists with an "optimized_" prefix if you renamed them
        model_path_fallback = model_path.replace("baseline_", "optimized_")
        try:
            clf = joblib.load(model_path_fallback)
        except FileNotFoundError:
            print(f"Error: {model_path} or {scaler_path} not found. Skipping...")
            return None

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*75)
    print(f"FULL CODEC TEST: {model_label.upper()} ({classifier_type.upper()}) | {codec_type.upper()}")
    print("="*75)
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n   [TN: {cm[0][0]}  FP: {cm[0][1]}]\n   [FN: {cm[1][0]}  TP: {cm[1][1]}]")
    print("-" * 75)

    # Return dictionary for saving
    return {
        "Model": model_label,
        "Codec": codec_type,
        "Classifier": classifier_type,
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1_Score": f1,
        "TN": cm[0][0],
        "FP": cm[0][1],
        "FN": cm[1][0],
        "TP": cm[1][1]
    }

if __name__ == "__main__":
    models = ["clap-laion-music", "mert-v0"]
    codecs = ["dac_processed", "encodec_processed"]
    classifiers = ["svm", "mlp", "knn", "random forest"]

    results_list = []

    for model in models:
        for codec in codecs:
            for clf_type in classifiers:
                result = run_full_codec_evaluation(model, codec, clf_type)
                if result:
                    results_list.append(result)

    # Logic to save the collected results
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv("codec_evaluation_results.csv", index=False)
        print(f"\n All results saved to codec_evaluation_results.csv")
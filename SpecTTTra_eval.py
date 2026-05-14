import os
import time
import torch
import librosa
import numpy as np
import pandas as pd
import json
import warnings
from tqdm import tqdm
from sonics import HFAudioClassifier
from sklearn.metrics import classification_report, confusion_matrix
from huggingface_hub import login
from pathlib import Path
from auc_roc_eval.py import evaluate_roc_auc, plot_roc_curve
from apply_transformations import apply_low_pass, apply_high_pass

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
login("your_hf_token_here")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASTER_TEST_FILE = "test.txt"


datasets = {
    # Original AI (Clean)
    "Suno_Clean": {"path": "/data/suno/audio", "label": "Fake", "ext": "*.mp3"},
    "Udio_Clean": {"path": "/data/udio/audio", "label": "Fake", "ext": "*.mp3"},
    
    # DAC Folders
    "Suno_DAC": {"path": "data_dac_ai/suno", "label": "Fake", "ext": "*.wav"},
    "Udio_DAC": {"path": "data_dac_ai/udio", "label": "Fake", "ext": "*.wav"},
    "Human_DAC": {"path": "data_dac_ai/lastfm", "label": "Real", "ext": "*.wav"},
    
    # EnCodec Mono 24k Folders
    "Suno_Enc24": {"path": "data_encodec_mono_24k/suno", "label": "Fake", "ext": "*.wav"},
    "Udio_Enc24": {"path": "data_encodec_mono_24k/udio", "label": "Fake", "ext": "*.wav"},
    "Human_Enc24": {"path": "data_encodec_mono_24k/lastfm", "label": "Real", "ext": "*.wav"}
}

MODEL_MAP = {
    "SpecTTTra-α-5s": "awsaf49/sonics-spectttra-alpha-5s",
    "SpecTTTra-β-5s": "awsaf49/sonics-spectttra-beta-5s",
    "SpecTTTra-γ-5s": "awsaf49/sonics-spectttra-gamma-5s",
    "SpecTTTra-α-120s": "awsaf49/sonics-spectttra-alpha-120s",
    "SpecTTTra-β-120s": "awsaf49/sonics-spectttra-beta-120s",
    "SpecTTTra-γ-120s": "awsaf49/sonics-spectttra-gamma-120s",
}

def process_audio_majority_vote(audio_path, model, max_time):
    try:

        # Load audio and resample to model's expected sample rate (if needed)
        audio, sr = librosa.load(audio_path, sr=16000)
        chunk_samples = int(max_time * sr)
        chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]
        chunks = [c for c in chunks if len(c) >= sr] 
        if not chunks: return None
        
        # Process each chunk through the model and collect probabilities
        chunk_probs = []
        for chunk in chunks:
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            with torch.no_grad():
                # Model expects a batch dimension, so we add one and move to device
                chunk_ts = torch.from_numpy(chunk).float().to(device).unsqueeze(0)
                pred = model(chunk_ts)

                # Sigmoid gives 0 (Real) to 1 (Fake)
                prob = torch.sigmoid(pred).item()
                chunk_probs.append(prob)
        
        # Use the mean probability as the "AI Score"
        ai_score = np.mean(chunk_probs)
        is_fake = ai_score > 0.5
        
        return {
            "Label": "Fake" if is_fake else "Real",
            "AI_Score": round(ai_score, 4), 
            "Total_Chunks": len(chunks)
        }
    except Exception: return None

def calc_and_append_metrics(model_name, config_name, data_subset, all_results_list):
    if not data_subset: return
    df = pd.DataFrame(data_subset)
    report = classification_report(df["True_Label"], df["Label"], output_dict=True, zero_division=0)
    fake_metrics = report.get('Fake', {'f1-score': 0, 'recall': 0, 'precision': 0})
    real_metrics = report.get('Real', {'f1-score': 0, 'recall': 0, 'precision': 0})
    
    tn, fp, fn, tp = confusion_matrix(df["True_Label"], df["Label"], labels=["Real", "Fake"]).ravel()
    summary = {
        "Model": f"{model_name} ({config_name})", 
        "Accuracy": round(report.get('accuracy', 0), 4), 
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Real_F1": round(real_metrics['f1-score'], 4), "Fake_F1": round(fake_metrics['f1-score'], 4),
    }
    all_results_list.append(summary)
    print(f"RESULTS FOR: {summary['Model']} | Accuracy: {summary['Accuracy']:.2%}")


if not os.path.exists(MASTER_TEST_FILE):
    raise FileNotFoundError(f"Missing {MASTER_TEST_FILE}")

with open(MASTER_TEST_FILE, "r") as f:
    master_test_ids = {Path(line.strip()).stem for line in f if line.strip()}

all_summary_results = []

for model_key, model_id in MODEL_MAP.items():
    print(f"\n{'='*90}\nEvaluating Model: {model_key}")
    model = HFAudioClassifier.from_pretrained(model_id).to(device).eval()
    max_time = model.config.audio.max_time
    
    clean_name = model_key.replace('-', '_').replace('α', 'alpha').replace('β', 'beta').replace('γ', 'gamma')
    json_log_path = f"progress_robustness_{clean_name}.json"
    
    progress_data = []
    if os.path.exists(json_log_path):
        with open(json_log_path, 'r') as f:
            progress_data = json.load(f)

    for d_name, info in datasets.items():
        base_path = Path(info["path"])
        if not base_path.exists(): continue
        
        all_files_on_disk = list(base_path.rglob(info["ext"]))
        test_files = [str(f) for f in all_files_on_disk if f.stem in master_test_ids]
        
        for f in tqdm(test_files, desc=f"  {d_name}"):
            if any(d["Track_ID"] == Path(f).stem and d["Dataset"] == d_name for d in progress_data):
                continue
            
            res = process_audio_majority_vote(f, model, max_time)
            if res:
                progress_data.append({"Track_ID": Path(f).stem, "True_Label": info["label"], "Dataset": d_name, **res})
                with open(json_log_path, 'w') as jf:
                    json.dump(progress_data, jf, indent=4)

    
    ai_dac = [d for d in progress_data if d["Dataset"] in ["Suno_DAC", "Udio_DAC"]]
    ai_enc24 = [d for d in progress_data if d["Dataset"] in ["Suno_Enc24", "Udio_Enc24"]]
    human_dac = [d for d in progress_data if d["Dataset"] == "Human_DAC"]
    human_enc24 = [d for d in progress_data if d["Dataset"] == "Human_Enc24"]

    calc_and_append_metrics(model_key, "DAC AI vs DAC Human", ai_dac + human_dac, all_summary_results)
    calc_and_append_metrics(model_key, "Enc24 AI vs Enc24 Human", ai_enc24 + human_enc24, all_summary_results)

    torch.cuda.empty_cache()

if all_summary_results:
    pd.DataFrame(all_summary_results).to_csv("sonics_codec_comparison.csv", index=False)
    print("\nSummary saved to 'sonics_codec_comparison.csv'")

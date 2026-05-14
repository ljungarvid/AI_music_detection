import os
import torch
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download
import laion_clap
from baseline import extract_clap_embeddings, extract_mert_embeddings 


cutoff_freqs = [0.1, 0.5, 1, 3, 5, 8, 10, 12, 16, 20] # kHz
sampling_rates = [8, 12, 16, 24, 48] # kHz 


def butter_lowpass(cutoff, fs, order=5):
    # Calculate Butterworth low-pass filter coefficients
    nyquist = 0.5 * fs
    # Ensure cutoff frequency is less than Nyquist to avoid instability
    normal_cutoff = min(cutoff / nyquist, 0.99)
    # Get filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    # Calculate Butterworth high-pass filter coefficients
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_low_pass(audio, sr, rel_path, output_dir, lp_cutoff, clap_model, mert_model, mert_proc, order=5):
    source_type = rel_path.split('/')[0].lower()
    file_name = Path(rel_path).stem
    freq_str = f"lp_{lp_cutoff/1000}khz"

    clap_file = output_dir / "clap-laion-music" / freq_str / source_type / f"{file_name}.npy"
    mert_file = output_dir / "mert-v0" / freq_str / source_type / f"{file_name}.npy"

    if clap_file.exists() and mert_file.exists():
        return 

    clap_file.parent.mkdir(parents=True, exist_ok=True)
    mert_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Retrieve low-pass Butterworth filter coefficients
    b_lp, a_lp = butter_lowpass(lp_cutoff, sr, order=order)
    # Apply low-pass filter to the audio
    y = filtfilt(b_lp, a_lp, audio)
    
    # Extract and save CLAP and MERT embeddings for the low-pass filtered audio
    y_clap = librosa.util.normalize(y)
    clap_emb = extract_clap_embeddings(clap_model, y_clap, 48000)
    np.save(clap_file, clap_emb)
    
    y_mert = librosa.resample(y, orig_sr=48000, target_sr=16000)
    y_mert = librosa.util.normalize(y_mert)
    mert_emb = extract_mert_embeddings(mert_model, mert_proc, y_mert, 16000)
    np.save(mert_file, mert_emb)

def apply_high_pass(audio, sr, rel_path, output_dir, hp_cutoff, clap_model, mert_model, mert_proc, order=5):
    source_type = rel_path.split('/')[0].lower()
    file_name = Path(rel_path).stem
    freq_str = f"hp_{hp_cutoff/1000}khz"

    clap_file = output_dir / "clap-laion-music" / freq_str / source_type / f"{file_name}.npy"
    mert_file = output_dir / "mert-v0" / freq_str / source_type / f"{file_name}.npy"

    if clap_file.exists() and mert_file.exists():
        return 

    clap_file.parent.mkdir(parents=True, exist_ok=True)
    mert_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Retrieve high-pass Butterworth filter coefficients
    b_hp, a_hp = butter_highpass(hp_cutoff, sr, order=order)
    # Apply high-pass filter to the audio
    y = filtfilt(b_hp, a_hp, audio)
    
    # Extract and save CLAP and MERT embeddings for the high-pass filtered audio
    y_clap = librosa.util.normalize(y)
    clap_emb = extract_clap_embeddings(clap_model, y_clap, 48000)
    np.save(clap_file, clap_emb)
    
    y_mert = librosa.resample(y, orig_sr=48000, target_sr=16000)
    y_mert = librosa.util.normalize(y_mert)
    mert_emb = extract_mert_embeddings(mert_model, mert_proc, y_mert, 16000)
    np.save(mert_file, mert_emb)

def simulate_resampling(audio, original_sr, attack_sr, rel_path, output_dir, clap_model, mert_model, mert_processor):
    source_type = rel_path.split('/')[0].lower()
    file_name = Path(rel_path).stem
    res_str = f"res_{attack_sr/1000}khz"

    clap_file = output_dir / "clap-laion-music" / res_str / source_type / f"{file_name}.npy"
    mert_file = output_dir / "mert-v0" / res_str / source_type / f"{file_name}.npy"

    if clap_file.exists() and mert_file.exists():
        return

    clap_file.parent.mkdir(parents=True, exist_ok=True)
    mert_file.parent.mkdir(parents=True, exist_ok=True)

    # Simulate resampling attack by downsampling and then upsampling back to 48kHz
    y_attack = librosa.resample(audio, orig_sr=original_sr, target_sr=attack_sr)
    
    # Resample for CLAP (48kHz)
    y_clap = librosa.resample(y_attack, orig_sr=attack_sr, target_sr=48000)
    y_clap = librosa.util.normalize(y_clap)
    clap_emb = extract_clap_embeddings(clap_model, y_clap, 48000)
    np.save(clap_file, clap_emb)
    
    # Resample for MERT (16kHz)
    y_mert = librosa.resample(y_attack, orig_sr=attack_sr, target_sr=16000)
    y_mert = librosa.util.normalize(y_mert)
    mert_emb = extract_mert_embeddings(mert_model, mert_processor, y_mert, 16000)
    np.save(mert_file, mert_emb)

if __name__ == "__main__":
    print("Loading models to 3090...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading CLAP...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
    clap_model.load_ckpt(clap_ckpt)
    clap_model.to(device).eval()

    print("Loading MERT...")
    mert_name = "m-a-p/MERT-v0"
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_name, trust_remote_code=True)
    mert_config = AutoConfig.from_pretrained(mert_name, trust_remote_code=True)
    if not hasattr(mert_config, 'conv_pos_batch_norm'): mert_config.conv_pos_batch_norm = False
    mert_model = AutoModel.from_pretrained(mert_name, config=mert_config, trust_remote_code=True).to(device).eval()

    MASTER_TEST_FILE = "test.txt"
    OUTPUT_DIR = Path("sampled_embeddings") 
    DATA_PREFIX = Path("/data") 

    if os.path.exists(MASTER_TEST_FILE):
        with open(MASTER_TEST_FILE, "r") as f:
            target_paths = [line.strip() for line in f if line.strip()]

        print(f"Checking for existing files in {OUTPUT_DIR}...")
        to_process = []
        for rel_path in target_paths:
            source_type = rel_path.split('/')[0].lower()
            track_id = Path(rel_path).stem
            proxy_file = OUTPUT_DIR / "clap-laion-music" / "res_8.0khz" / source_type / f"{track_id}.npy"
            if not proxy_file.exists():
                to_process.append(rel_path)

        print(f"Remaining to process: {len(to_process)}")

        for rel_path in tqdm(to_process, desc="Running Transformations"):
            input_path = DATA_PREFIX / rel_path
            if not input_path.exists():
                continue
            
            try:
                # Load master audio at 48k for filtering
                audio, sr_orig = librosa.load(input_path, sr=48000)
                
                
                for freq in cutoff_freqs:
                    apply_low_pass(audio, sr_orig, rel_path, OUTPUT_DIR, freq * 1000, clap_model, mert_model, mert_processor)
                    apply_high_pass(audio, sr_orig, rel_path, OUTPUT_DIR, freq * 1000, clap_model, mert_model, mert_processor)
                
                for s_khz in sampling_rates:
                    simulate_resampling(audio, 48000, s_khz*1000, rel_path, OUTPUT_DIR, clap_model, mert_model, mert_processor)

                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError processing {rel_path}: {e}")

    print(f"\nProcessing complete. Embeddings saved in: {OUTPUT_DIR}")
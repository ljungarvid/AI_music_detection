import os
import time
import torch
import numpy as np
import librosa
import laion_clap
import warnings
from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, AutoModel, AutoConfig
from huggingface_hub import hf_hub_download, login


os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
login("your_hf_token_here")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")


def extract_clap_embeddings(model, audio_input, sr=48000):
    """
    audio_input: Can be a file path (str/Path) OR a numpy array.
    """
    try:
        # Check if input is a path 
        if isinstance(audio_input, (str, Path)):
            audio, sr = librosa.load(audio_input, sr=48000, mono=True)
        else:
            
            audio = audio_input
            if sr != 48000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
                sr = 48000

        chunk_samples = 10 * sr
        num_chunks = min(len(audio) // chunk_samples, 18) 
        if num_chunks == 0: return None
        
        # Process the audio in chunks to manage memory usage, then average the embeddings
        song_chunks = np.stack([audio[j*chunk_samples:(j+1)*chunk_samples] for j in range(num_chunks)])
        chunk_tensor = torch.from_numpy(song_chunks).float().to(device)

        with torch.inference_mode():
            # Get embeddings for each chunk and average them to get a single embedding for the whole song
            chunk_embeddings = model.get_audio_embedding_from_data(chunk_tensor, use_tensor=True)
            # Average over the chunk dimension to get a single embedding per song, then move to CPU and convert to numpy
            return chunk_embeddings.mean(dim=0).cpu().numpy().astype(np.float32)
            
    except Exception as e:
        print(f"Error CLAP: {e}")
        torch.cuda.empty_cache()
        return None

def extract_mert_embeddings(model, processor, audio_input, sr=16000):
    """
    audio_input: Can be a file path (str/Path) OR a numpy array.
    """
    try:
        if isinstance(audio_input, (str, Path)):
            waveform, sr = librosa.load(audio_input, sr=16000, mono=True)
        else:
            waveform = audio_input
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                sr = 16000
        
        chunk_samples = 5 * sr
        chunks = [waveform[j:j+chunk_samples] for j in range(0, len(waveform), chunk_samples) 
                  if len(waveform[j:j+chunk_samples]) >= sr]
        if not chunks: return None

        sub_batch_size = 4
        song_chunk_embs = []
        # Process chunks in sub-batches to manage memory usage, then average the embeddings
        for k in range(0, len(chunks), sub_batch_size):
            sub_chunks = chunks[k : k + sub_batch_size]
            inputs = processor(sub_chunks, sampling_rate=sr, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1] 
                # Average over time dimension to get a single embedding per chunk, then move to CPU and convert to numpy
                batch_embs = hidden.mean(dim=1).cpu().numpy()
                song_chunk_embs.append(batch_embs)
        # Average the chunk embeddings to get a single embedding for the whole song
        return np.mean(np.vstack(song_chunk_embs), axis=0).astype(np.float32)
        
    except Exception as e:
        print(f"Error MERT: {e}")
        torch.cuda.empty_cache()
        return None

def load_processed_paths(root_dir):
    root = Path(root_dir)
    return [str(p) for p in root.rglob('*') if p.suffix in ['.wav', '.mp3']]



if __name__ == "__main__":
    start_total = time.time()

    # Load CLAP Once
    print("Loading CLAP...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
    clap_model.load_ckpt(clap_ckpt)
    clap_model.to(device).eval()

    # Load MERT Once
    print("Loading MERT...")
    mert_name = "m-a-p/MERT-v0"
    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(mert_name, trust_remote_code=True)
    mert_config = AutoConfig.from_pretrained(mert_name, trust_remote_code=True)
    if not hasattr(mert_config, 'conv_pos_batch_norm'): mert_config.conv_pos_batch_norm = False
    mert_model = AutoModel.from_pretrained(mert_name, config=mert_config, trust_remote_code=True).to(device).eval()

    configs = [
        {"root": "data_dac_ai", "label": "dac_processed"},
        {"root": "data_encodec_mono_24k", "label": "encodec_processed"}
    ]

    for cfg in configs:
        print(f"\nProcessing {cfg['label']}...")
        paths = sorted(load_processed_paths(cfg['root']))
        
        if paths:
            extract_clap_embeddings(clap_model, paths, cfg['root'], cfg['label'])
            extract_mert_embeddings(mert_model, mert_processor, paths, cfg['root'], cfg['label'])

    print(f"\nTotal Time: {time.time() - start_total:.2f}s")
import os
import torch
import dac
from audiotools import AudioSignal
from pathlib import Path
from tqdm import tqdm

os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the DAC model (44.1kHz version)
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to(device).eval()

MASTER_TEST_FILE = "test_clap-laion-music.txt"
OUTPUT_DIR = Path("data_dac_ai")

if not os.path.exists(MASTER_TEST_FILE):
    raise FileNotFoundError(f"Missing {MASTER_TEST_FILE}")

with open(MASTER_TEST_FILE, "r") as f:
    all_lines = [line.strip() for line in f if line.strip()]

target_paths = list(set([
    line for line in all_lines 
    if line.startswith("suno/") or line.startswith("udio/") or line.startswith("lastfm/")
]))

def process_dac(in_p, out_p, chunk_duration=10.0):
    try:

        # Load audio, convert to mono, resample to 44.1kHz if needed
        sig = AudioSignal(in_p)
        if sig.num_channels > 1: sig.to_mono()
        if sig.sample_rate != 44100: sig.resample(44100)
        
        sr = sig.sample_rate
        audio_data = sig.audio_data  
        total_samples = audio_data.shape[-1]
        # Process in chunks to manage memory usage
        samples_per_chunk = int(chunk_duration * sr)
        reconstructed_chunks = []
        
        # For each chunk, run through the DAC model and store the reconstructed audio
        for start_sample in range(0, total_samples, samples_per_chunk):
            end_sample = min(start_sample + samples_per_chunk, total_samples)
            # Skip very short chunks at the end
            if end_sample - start_sample < 100: continue
            
            # Prepare input for the model
            input_wav = audio_data[..., start_sample:end_sample].to(device)
            with torch.inference_mode():
                x = model.preprocess(input_wav, sr)
                # Encode and decode the chunk
                z, _, _, _, _ = model.encode(x)
                y = model.decode(z)
            # Store the reconstructed chunk
            reconstructed_chunks.append(y.cpu())
            
            del input_wav, x, z, y
            torch.cuda.empty_cache()

        if reconstructed_chunks:
            # Concatenate all reconstructed chunks and save the full audio
            full_tensor = torch.cat(reconstructed_chunks, dim=-1)
            full_reconstructed = AudioSignal(full_tensor, sample_rate=sr)
            full_reconstructed.write(out_p)
            return True
        return False
    except Exception as e:
        tqdm.write(f"\n ERROR Failed {in_p}: {e}")
        return False


print(f"Checking for existing files in {OUTPUT_DIR}...")


to_process = []
for rel_path in target_paths:
    target_path = (OUTPUT_DIR / rel_path).with_suffix('.wav')
    if not target_path.exists():
        to_process.append(rel_path)

skipped = len(target_paths) - len(to_process)
print(f"Skipped {skipped} files. Remaining to process: {len(to_process)}")

if to_process:
    success_count = 0
    for rel_path in tqdm(to_process, desc="Running DAC"):
        input_path = os.path.join("/data", rel_path)
        target_path = (OUTPUT_DIR / rel_path).with_suffix('.wav')
        
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(input_path):
            continue
            
        if process_dac(input_path, str(target_path)):
            success_count += 1

    print(f"\nDone. Processed {success_count} new tracks.")
else:
    print("\nNo new tracks to process.")
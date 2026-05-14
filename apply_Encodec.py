import os
import torch
from encodec import EncodecModel
from audiotools import AudioSignal
from pathlib import Path
from tqdm import tqdm

os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the EnCodec model (24kHz version)
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0) 
model.to(device).eval()

MASTER_TEST_FILE = "test_clap-laion-music.txt"
OUTPUT_DIR = Path("data_encodec_mono_24k")

if not os.path.exists(MASTER_TEST_FILE):
    raise FileNotFoundError(f"Missing {MASTER_TEST_FILE}")


with open(MASTER_TEST_FILE, "r") as f:
    all_paths = [line.strip() for line in f if line.strip()]
all_paths = list(set(all_paths))

def process_encodec_24k(in_p, out_p, chunk_sec=10.0):
    try:
        sig = AudioSignal(in_p)

        # Convert to mono and resample to 24kHz if needed
        if sig.num_channels > 1:
            sig.to_mono()
        if sig.sample_rate != 24000:
            sig.resample(24000)

        
        sr = 24000
        stride = 320 
        total_samples = sig.audio_data.shape[-1]
        samples_per_chunk = int(chunk_sec * sr)
        
        # Ensure chunk size is a multiple of stride to prevent alignment issues
        samples_per_chunk = (samples_per_chunk // stride) * stride
        
        reconstructed_chunks = []

        # Process in chunks to manage memory usage
        for start in range(0, total_samples, samples_per_chunk):
            end = min(start + samples_per_chunk, total_samples)
            chunk = sig.audio_data[..., start:end]
            
            # Apply padding to the last chunk if it's not a multiple of the stride
            curr_len = chunk.shape[-1]
            pad_needed = (stride - (curr_len % stride)) % stride
            
            # Prepare the chunk for the model
            wav_tensor = chunk.to(device).contiguous()
            if pad_needed > 0:
                wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_needed))

            with torch.inference_mode():
                # Encode and decode the chunk
                encoded_frames = model.encode(wav_tensor)
                y_chunk = model.decode(encoded_frames)
                # Remove padding from this chunk
                y_chunk = y_chunk[..., :curr_len]
                reconstructed_chunks.append(y_chunk.cpu())

        # Concatenate all reconstructed chunks and save the full audio
        if reconstructed_chunks:
            full_tensor = torch.cat(reconstructed_chunks, dim=-1)
            full_reconstructed = AudioSignal(full_tensor, sample_rate=24000)
            full_reconstructed.write(out_p)
            return True
        return False

    except Exception as e:
        tqdm.write(f"\n ERROR Failed {in_p}: {e}")
        return False


print(f"Applying Chunked Mono EnCodec 24kHz to {len(all_paths)} tracks...")

success_count = 0
error_count = 0

for rel_path in tqdm(all_paths):
    input_path = os.path.join("/data", rel_path)
    target_path = OUTPUT_DIR / rel_path
    target_path = target_path.with_suffix('.wav')
    
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        if not os.path.exists(input_path):
            continue
            
        if process_encodec_24k(input_path, str(target_path)):
            success_count += 1
        else:
            error_count += 1

print(f"\nComplete! Processed {len(all_paths)} tracks.")
print(f"Success: {success_count} | Errors: {error_count}")
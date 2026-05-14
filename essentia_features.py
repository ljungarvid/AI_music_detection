import essentia.standard as es
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Setup input paths
DATA_PATHS = {
    "suno": "/data/suno/audio",
    "udio": "/data/udio/audio",
    "lastfm": "/data/lastfm"
}
SPLIT_FILES = ["train.txt", "val.txt", "test.txt"] 

def collect_audio_files():
    """
    Collects specific LastFM files from txt splits and 
    all Suno/Udio files from their respective directories.
    """
    all_tasks = []

    # 1. Process LastFM (Filtered by your .txt split files)
    print("Gathering LastFM files from splits...")
    for txt_file in SPLIT_FILES:
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    audio_id = line.strip()
                    if audio_id:
                        path = os.path.join(DATA_PATHS["lastfm"], "audio", f"{audio_id}.mp3")
                        if os.path.exists(path):
                            all_tasks.append((path, "lastfm"))

    # 2. Process Suno and Udio (Full folder extraction)
    for label in ["suno", "udio"]:
        print(f"Gathering all files from {label} folder...")
        folder_path = DATA_PATHS[label]
        if os.path.exists(folder_path):
            files = list(Path(folder_path).glob("*.mp3"))
            for f in files:
                all_tasks.append((str(f), label))
                
    return all_tasks

def process_file(args):
    """
    Worker function for Essentia extraction.
    """
    audio_path, label = args
    # Output structure: essentia_features/label/hash.json
    output_dir = os.path.join("essentia_features", label)
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}.json")

    # Skip if already processed
    if os.path.exists(json_path):
        return

    try:
        # Extract features (CPU intensive)
        features, _ = es.MusicExtractor(
            lowlevelStats=['mean', 'stdev'],
            rhythmStats=['mean', 'stdev'],
            tonalStats=['mean', 'stdev']
        )(audio_path)

        # Save as JSON
        es.YamlOutput(filename=json_path, format='json')(features)
    except Exception:
        # Silently skip errors (corrupted files, etc.)
        pass

if __name__ == "__main__":
    # 1. Gather all files to be processed
    tasks = collect_audio_files()
    print(f"Total files identified for extraction: {len(tasks)}")

    if tasks:
        # 2. Parallel processing using all available CPU cores
        print(f"Starting Essentia extraction on {os.cpu_count()} cores...")
        with ProcessPoolExecutor() as executor:
            # We pass the full task list to tqdm for a master progress bar
            list(tqdm(executor.map(process_file, tasks),
                      total=len(tasks),
                      desc="Essentia Extraction"))

    print("\nAll Essentia features extracted to 'essentia_features/'.")
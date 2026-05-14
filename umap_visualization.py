import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_organized_embeddings(model_name):
    """
    Loads embeddings from the consolidated 'embeddings' directory.
    """
    X = []
    labels = []
    paths = []
    
    label_map = {
        "lastfm": "Human (LastFM)",
        "suno": "AI (Suno)",
        "udio": "AI (Udio)"
    }

    base_dir = "embeddings"

    for folder, display_name in label_map.items():
        folder_path = os.path.join(base_dir, model_name, folder)
        
        if not os.path.exists(folder_path):
            print(f"Skipping {folder}, path not found: {folder_path}")
            continue
            
        count = 0
        for npy_file in Path(folder_path).glob("*.npy"):
            emb = np.load(npy_file)
            if np.isnan(emb).any():
                continue
                
            X.append(emb)
            labels.append(display_name)
            paths.append(str(npy_file)) 
            count += 1
        print(f"Loaded {count} embeddings for {display_name} from: {folder_path}")

    return np.array(X), labels, paths

def create_umap_plot(X, labels, title, save_path):
    print(f"Running UMAP for {title}...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    umap_model = UMAP(
        n_neighbors=15,      
        min_dist=0.5,        
        metric='manhattan',
        random_state=42 
    )
    
    X_umap = umap_model.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    palette = {'Human (LastFM)': '#2ca02c', 'AI (Suno)': '#1f77b4', 'AI (Udio)': '#d62728'}
    
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, 
        palette=palette, s=20, alpha=0.3, edgecolor='none'
    )
    
    plt.title(f'UMAP Visualization: {title}', fontsize=15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Successfully saved plot to {save_path}")
    
    return X_umap 


def calculate_similarity(idx1, idx2, embeddings):
    """Computes cosine similarity between two specific embeddings."""
    emb1 = embeddings[idx1].reshape(1, -1)
    emb2 = embeddings[idx2].reshape(1, -1)
    score = cosine_similarity(emb1, emb2)[0][0]

    return score

def get_url_from_json(label, track_path):
    import json
    from pathlib import Path
    
    track_id = Path(track_path).stem
    
    if "LastFM" in label:
        data_folder = "lastfm"
    elif "Suno" in label:
        data_folder = "suno"
    elif "Udio" in label:
        data_folder = "udio"
    else:
        return "Unknown Dataset"

    json_path = Path("/data") / data_folder / "metadata" / f"{track_id}.json"
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                # replace NaN to avoid JSON decode errors if they exist
                content = f.read().replace('NaN', 'null')
                data = json.loads(content)
                
                if data_folder == "lastfm":
                    return data.get('url_lastfm', 'url_lastfm key not found')
                
                return data.get('audio_url', 'audio_url key not found')
                
        except Exception as e:
            return f"Error reading JSON: {e}"
    
    return f"JSON not found at: {json_path}"

def edge_analysis(X_embeddings, X_umap, labels, paths, model_name):
    """
    Finds the 2 tracks at each extreme edge of the UMAP plot and 
    computes their internal similarity and metadata URLs.
    """
    # Identify indices for the 2 tracks at each edge
    edges = {
        "LEFT EDGE": np.argsort(X_umap[:, 0])[:2],
        "RIGHT EDGE": np.argsort(X_umap[:, 0])[-2:],
        "TOP EDGE": np.argsort(X_umap[:, 1])[-2:],
        "BOTTOM EDGE": np.argsort(X_umap[:, 1])[:2]
    }

    print(f"\nDOUBLE-EDGE SIMILARITY AUDIT ({model_name}) \n")

    for edge_name, indices in edges.items():
        print(f"--- {edge_name} ---")
        idx1, idx2 = indices[0], indices[1]
        
        # Compute similarity between the two tracks at this specific edge
        sim = cosine_similarity(X_embeddings[idx1].reshape(1,-1), X_embeddings[idx2].reshape(1,-1))[0][0]
        
        for i, idx in enumerate([idx1, idx2]):
            label = labels[idx]
            path = paths[idx]
            audio_url = get_url_from_json(label, path)
            
            print(f"  Track {i+1} ({label}):")
            print(f"    ID:  {Path(path).stem}")
            print(f"    URL: {audio_url}")
                
        print(f" similarity score: {sim:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    
    X_clap, labels_clap, paths_clap = load_organized_embeddings("clap-laion-music")
    if len(X_clap) > 0:
        X_umap_clap = create_umap_plot(X_clap, labels_clap, "CLAP Embeddings", "umap_clap.png")
        edge_analysis(X_clap, X_umap_clap, labels_clap, paths_clap, "CLAP")

    X_mert, labels_mert, paths_mert = load_organized_embeddings("mert-v0")
    if len(X_mert) > 0:
        X_umap_mert = create_umap_plot(X_mert, labels_mert, "MERT Embeddings", "umap_mert.png")
        edge_analysis(X_mert, X_umap_mert, labels_mert, paths_mert, "MERT")
    
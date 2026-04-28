from turtle import pd

import numpy as np
import joblib
import json
from pathlib import Path
import hashlib


class FeatureStore:
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.base_path / "metadata.json"
        
        if not self.meta_path.exists():
            self._save_metadata({})
    
    # ------------------------
    # Metadata Handling
    # ------------------------
    
    def _load_metadata(self):
        with open(self.meta_path, "r") as f:
            return json.load(f)
    
    def _save_metadata(self, meta):
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    # ------------------------
    # Embeddings
    # ------------------------
    
    def save_embeddings(self, name: str, features: np.ndarray):
        path = self.base_path / f"{name}.npy"
        np.save(path, features)
        
        meta = self._load_metadata()
        meta[name] = {"type": "embedding", "shape": features.shape}
        self._save_metadata(meta)
        
        print(f"Saved embeddings → {path}")
    
    def load_embeddings(self, name: str):
        path = self.base_path / f"{name}.npy"
        if not path.exists():
            return None
        
        print(f"Loaded embeddings ← {path}")
        return np.load(path)
    
    # ------------------------
    # TF-IDF
    # ------------------------
    
    def save_tfidf(self, name: str, pipeline):
        path = self.base_path / f"{name}.joblib"
        joblib.dump(pipeline, path)
        
        meta = self._load_metadata()
        meta[name] = {"type": "tfidf"}
        self._save_metadata(meta)
        
        print(f"Saved TF-IDF → {path}")
    
    def load_tfidf(self, name: str):
        path = self.base_path / f"{name}.joblib"
        if not path.exists():
            return None
        
        print(f"Loaded TF-IDF ← {path}")
        return joblib.load(path)
    
    
    
    def _load_metadata(self):
        with open(self.meta_path, "r") as f:
            return json.load(f)
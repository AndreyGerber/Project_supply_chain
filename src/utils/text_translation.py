import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Dict

from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator


# ================================
# 🔒 DETERMINISMUS
# ================================

DetectorFactory.seed = 42  # wichtig: langdetect sonst random!


# ================================
# 📦 CACHE
# ================================

class TranslationCache:
    def __init__(self, cache_path: str = "data/translation_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str):
        return self.cache.get(self._hash(text))

    def set(self, text: str, translation: str):
        self.cache[self._hash(text)] = translation


# ================================
# 🌍 LANGUAGE DETECTION
# ================================

def detect_lang(text: str) -> str:
    try:
        if not isinstance(text, str) or len(text.strip()) < 3:
            return "unknown"
        return detect(text)
    except Exception:
        return "unknown"


# ================================
# 🌐 TRANSLATOR CLASS
# ================================

class ReviewTranslator:
    def __init__(
        self,
        target_lang: str = "en",
        batch_size: int = 32,
        cache_path: str = "data/translation_cache.json"
    ):
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.translator = GoogleTranslator(source="auto", target=target_lang)
        self.cache = TranslationCache(cache_path)

    # ----------------------------
    # 🔁 SINGLE TRANSLATION
    # ----------------------------
    def translate_text(self, text: str) -> str:
        if pd.isna(text) or text.strip() == "":
            return text

        # Cache check
        cached = self.cache.get(text)
        if cached:
            return cached

        try:
            translated = self.translator.translate(text)
            self.cache.set(text, translated)
            return translated
        except Exception:
            return text  # fallback

    # ----------------------------
    # ⚡ BATCH TRANSLATION
    # ----------------------------
    def translate_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.translate_text(text))
        return results

    # ----------------------------
    # 🧠 MAIN PIPELINE
    # ----------------------------
    def process_dataframe_for_translation(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        print("🌍 Detecting languages...")
        df["language"] = df["review_text"].astype(str).apply(detect_lang)

        # Split
        df_en = df[df["language"] == "en"].copy()
        df_not_en = df[df["language"] != "en"].copy()

        print(f"English: {len(df_en)}, Non-English: {len(df_not_en)}")

        # Batch translation
        translated_texts = []

        texts = df_not_en["review_text"].tolist()

        print("🌐 Translating in batches...")

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            translated_batch = self.translate_batch(batch)
            translated_texts.extend(translated_batch)

        df_not_en["review_text_en"] = translated_texts

        # English stays unchanged
        df_en["review_text_en"] = df_en["review_text"]

        # Merge + order
        df_final = pd.concat([df_en, df_not_en], axis=0)
        df_final = df_final.sort_index()

        # Save cache after run
        self.cache.save()

        print("✅ Translation complete & cached.")

        return df_final
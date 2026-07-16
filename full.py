"""
Tanglish -> Tamil -> English pipeline
Step 1: ai4bharat XlitEngine transliterates Tanglish (Latin script) to Tamil script
Step 2: facebook/nllb-200-distilled-600M translates Tamil script to English
"""

import os

# --- HF cache / logging config (must run before transformers import) ---
os.environ["HF_HUB_CACHE"] = "./model"
os.environ["HF_HUB_DISABLE_SYSLOG_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# --- START MONKEY PATCH (omegaconf compatibility fix for ai4bharat) ---
from omegaconf import OmegaConf

original_structured = OmegaConf.structured

def safe_structured(obj, *args, **kwargs):
    try:
        return original_structured(obj, *args, **kwargs)
    except Exception:
        return obj

OmegaConf.structured = safe_structured
# --- END MONKEY PATCH ---

from ai4bharat.transliteration import XlitEngine
from transformers import pipeline


class TanglishToEnglishPipeline:
    def __init__(self, beam_width: int = 10):
        print("Loading transliteration engine (Tanglish -> Tamil)...")
        self.xlit_engine = XlitEngine("ta", beam_width=beam_width, rescore=True)

        print("Loading translation engine (Tamil -> English)...")
        self.translate_engine = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang="tam_Taml",
            tgt_lang="eng_Latn",
        )
        print("Pipeline ready!\n")

    def transliterate(self, text: str) -> str:
        """Tanglish (Latin script) -> Tamil script"""
        result = self.xlit_engine.translit_sentence(text.strip())
        return result["ta"]

    def translate(self, tamil_text: str) -> str:
        """Tamil script -> English"""
        raw_output = self.translate_engine(tamil_text)
        return raw_output[0]["translation_text"]

    def run(self, tanglish_text: str) -> dict:
        """Full pipeline: Tanglish -> Tamil -> English"""
        tamil_text = self.transliterate(tanglish_text)
        english_text = self.translate(tamil_text)

        return {
            "tanglish_input": tanglish_text.strip(),
            "tamil_transliteration": tamil_text,
            "english_translation": english_text,
        }


if __name__ == "__main__":
    pipeline_obj = TanglishToEnglishPipeline(beam_width=10)

    test_phrases = [
        "enna saappida poren",
        "neenga enna seyyareenga",
        "naan velaikku pogiren",
        "neenga eppo varuveenga",
        "vaanga veliya povam",
    ]

    print("--- Running Full Tanglish -> Tamil -> English Pipeline ---\n")
    for phrase in test_phrases:
        try:
            result = pipeline_obj.run(phrase)
            print(f"Tanglish: {result['tanglish_input']}")
            print(f"Tamil:    {result['tamil_transliteration']}")
            print(f"English:  {result['english_translation']}")
            print("-" * 40)
        except Exception as e:
            print(f"Failed to process '{phrase}': {e}")
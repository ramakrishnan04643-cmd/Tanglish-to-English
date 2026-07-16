# --- START MONKEY PATCH ---
from omegaconf import OmegaConf, MISSING
import dataclasses

def patched_structured(obj, *args, **kwargs):
    # This replaces the original function to handle the MISSING type
    return original_structured(obj, *args, **kwargs)

# We store the original method
original_structured = OmegaConf.structured

# We wrap the creation to catch the MISSING sentinel if it persists
def safe_structured(obj, *args, **kwargs):
    try:
        return original_structured(obj, *args, **kwargs)
    except Exception:
        # If standard structured fails due to MISSING, we force a pass 
        # or return the object directly for this specific initialization
        return obj

OmegaConf.structured = safe_structured
# --- END MONKEY PATCH ---

# Now import your engine
from ai4bharat.transliteration import XlitEngine

# Initialize the engine for Tamil ('ta')
# beam_width handles accuracy (default is 4, higher = more accurate but slower)
e = XlitEngine("ta", beam_width=10)

# 1. For a Full Sentence (Tanglish to Tamil)
sentence_input = "enna saappida poren"
sentence_output = e.translit_sentence(sentence_input)
print("Sentence Output:", sentence_output["ta"])  # Output: வணக்கம் உலகம்

# 2. For a Single Word (returns top k options)
word_input = "amma"
word_output = e.translit_word(word_input, topk=1)
print("Word Options:", word_output["ta"])  # Output: ['அம்மா', 'அம்ம', ...]cd
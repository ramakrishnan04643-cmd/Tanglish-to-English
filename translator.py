"""
Tanglish -> English pipeline (v3 - single-step, Hugging Face)
One LLM call does both jobs at once: produces the Tamil script AND the English
translation together, using full sentence context. This avoids compounding
errors from a two-stage transliterate-then-translate pipeline.
"""

import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the same folder, sets env vars from it

# --- Hugging Face Inference API config ---
# Get a free token at https://huggingface.co/settings/tokens
# Token needs "Make calls to Inference Providers" permission enabled.
# Put it in your .env file as: HF_API_TOKEN=hf_xxxxx
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
LLM_MODEL = "meta-llama/Llama-3.3-70B-Instruct:groq"  # groq's LPU hardware = fast, reliable latency


class TanglishToEnglishPipeline:
    def __init__(self):
        if not HF_API_TOKEN:
            raise RuntimeError(
                "HF_API_TOKEN environment variable not set. "
                "Get a token at https://huggingface.co/settings/tokens, "
                "and put it in a .env file as HF_API_TOKEN=hf_xxxxx"
            )
        print("Pipeline ready (single-step LLM mode via Hugging Face)!\n")

    def run(self, tanglish_text: str) -> dict:
        """
        Full pipeline in one LLM call: Tanglish -> Tamil script + English translation.
        Returns both together so the model can use full sentence context for each,
        rather than passing a possibly-wrong intermediate result to a second model.
        """
        tanglish_text = tanglish_text.strip()

        prompt = (
            "You are an expert Tanglish translator. Tanglish is Tamil written "
            "using English/Latin letters, often mixed with genuine English words, "
            "with inconsistent, colloquial spelling.\n\n"
            "IMPORTANT disambiguation rules:\n"
            "- Not every capitalized-sounding or name-shaped word is a person's name. "
            "Common adverbs like 'sikirama'/'seekiramaa' (quickly), 'konjam' (a little), "
            "'nalla' (well/good) are frequently mistaken for names when romanized - "
            "check if the sentence makes more sense with it as a normal word first.\n"
            "- Words like 'da', 'pa', 'machi', 'di' at the end of a sentence are casual "
            "address particles (like 'dude'/'man') - they add tone, not new meaning or "
            "a new person being addressed.\n"
            "- A sentence rarely has more than one named person unless clearly indicated "
            "by structure (e.g. 'X sollu Y ku' = 'X, tell Y').\n"
            "- Short colloquial contractions of pronouns are common and easy to misread: "
            "'yen'/'yaen' before a verb can mean 'ennai' (me) rather than 'why' (ஏன்) - "
            "check whether 'why' already appears elsewhere in the sentence; if so, a second "
            "'yaen' is very likely the pronoun 'me', not a repeated 'why'.\n"
            "- Preserve verb tense exactly. Tamil past-tense markers (e.g. '-ta'/'-tu' endings "
            "like 'pota', 'senja', 'sonna') must stay simple past in English ('did X', 'picked', "
            "'said') - do not soften them into present continuous ('is doing', 'is picking').\n\n"
            "Example 1:\n"
            "Tanglish: prabhu sikirama officeku kelmbi vaa da\n"
            '{"tamil_script": "பிரபு சீக்கிரமா ஆபீஸுக்கு கிளம்பி வா டா", '
            '"english_translation": "Prabhu, get going quickly and come to the office, man."}\n\n'
            "Example 2:\n"
            "Tanglish: nee enna panra machi\n"
            '{"tamil_script": "நீ என்ன பண்ற மச்சி", '
            '"english_translation": "What are you doing, buddy?"}\n\n'
            "Example 3:\n"
            "Tanglish: ethku da yaen kuda sandai pota\n"
            '{"tamil_script": "எதுக்கு டா என்னோட சண்டை போட்டா", '
            '"english_translation": "Why did you pick a fight with me, man?"}\n\n'
            "Now translate the following, using the same disambiguation approach:\n\n"
            "Given the Tanglish sentence below, produce:\n"
            "1. tamil_script: the sentence rewritten in proper Tamil (Unicode) script, "
            "using full sentence context to resolve ambiguous or colloquial spellings.\n"
            "2. english_translation: a natural, fluent English translation of the sentence's "
            "actual meaning - not a word-for-word gloss.\n\n"
            "Respond with ONLY a raw JSON object, no markdown fences, no explanation, "
            "in exactly this shape:\n"
            '{"tamil_script": "...", "english_translation": "..."}\n\n'
            f"Tanglish sentence: {tanglish_text}"
        )

        max_attempts = 3
        last_error = None
        response = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    HF_ROUTER_URL,
                    headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
                    json={
                        "model": LLM_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1200,  # headroom for long/technical sentences
                        "temperature": 0.1,  # low temperature = consistent, literal output
                    },
                    timeout=60,  # 70B model can be slow under load; give it room
                )
                if response.status_code >= 500:
                    # Transient server-side issue (e.g. 504 Gateway Timeout) - worth retrying
                    print(
                        f"DEBUG - Server error {response.status_code} on attempt "
                        f"{attempt}/{max_attempts}, retrying..."
                    )
                    last_error = requests.exceptions.HTTPError(
                        f"{response.status_code} server error", response=response
                    )
                    if attempt < max_attempts:
                        continue
                break
            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"DEBUG - Timeout on attempt {attempt}/{max_attempts}, retrying...")
                if attempt == max_attempts:
                    raise TimeoutError(
                        f"HF API timed out after {max_attempts} attempts (60s each). "
                        f"The model may be under heavy load - try again shortly."
                    ) from last_error

        if response is not None and response.status_code >= 500:
            raise RuntimeError(
                f"HF API returned server errors on all {max_attempts} attempts "
                f"(last: {response.status_code}). The provider may be down - try again shortly."
            ) from last_error

        if not response.ok:
            print(f"DEBUG - HF API error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"No choices returned from LLM. Full response: {data}")

        message = choices[0].get("message") or {}
        raw_content = message.get("content")
        finish_reason = choices[0].get("finish_reason")

        if not raw_content:
            raise ValueError(f"LLM returned no usable content. Full message: {message}")

        if finish_reason == "length":
            raise ValueError(
                "LLM response was cut off before finishing (hit max_tokens limit). "
                "The sentence may be too long/complex for the current token budget. "
                f"Raw (truncated) output: {raw_content}"
            )

        raw_content = raw_content.strip()
        # Strip markdown code fences if the model added them despite instructions
        raw_content = re.sub(r'^```(?:json)?\s*|\s*```$', "", raw_content).strip()

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not parse LLM output as JSON: {e}. Raw output: {raw_content}"
            )

        tamil_text = parsed.get("tamil_script", "").strip()
        english_text = parsed.get("english_translation", "").strip()

        return {
            "tanglish_input": tanglish_text,
            "tamil_transliteration": tamil_text,
            "english_translation": english_text,
        }


# --- Module-level singleton so Flask app.py can just do:
#     from translator import translate_tanglish
#     translate_tanglish("prabhu naliku aven lover kuda veliya poron")
_pipeline_instance = None


def _get_pipeline() -> TanglishToEnglishPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TanglishToEnglishPipeline()
    return _pipeline_instance


def translate_tanglish(text: str) -> dict:
    """
    Entry point for app.py.
    Tanglish text in -> dict with tanglish_input, tamil_transliteration, english_translation
    """
    return _get_pipeline().run(text)


if __name__ == "__main__":
    pipeline_obj = TanglishToEnglishPipeline()

    test_phrases = [
        "prabhu sikirama office ku va da",
        "naliku enga povam",
        "enna saappida poren",
        "naan velaikku pogiren",
    ]

    print("--- Running Single-Step Tanglish -> English Pipeline (Hugging Face) ---\n")
    for phrase in test_phrases:
        try:
            result = pipeline_obj.run(phrase)
            print(f"Tanglish: {result['tanglish_input']}")
            print(f"Tamil:    {result['tamil_transliteration']}")
            print(f"English:  {result['english_translation']}")
            print("-" * 40)
        except Exception as e:
            print(f"Failed to process '{phrase}': {e}")
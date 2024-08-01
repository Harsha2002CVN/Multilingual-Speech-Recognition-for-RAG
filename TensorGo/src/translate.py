from transformers import MarianMTModel, MarianTokenizer

# Load MarianMT model for translation
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")

def translate_text(text, target_language="en"):
    translated = translation_model.generate(**translation_tokenizer.prepare_seq2seq_batch([text], return_tensors="pt"))
    return translation_tokenizer.decode(translated[0], skip_special_tokens=True)

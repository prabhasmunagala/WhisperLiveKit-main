
import logging
import sys
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_nllb():
    print("-" * 50)
    print("Testing NLLB functionality [VERSION 2 - CHECKING FLUSH]...")
    
    try:
        from nllw import load_model, OnlineTranslation
        from whisperlivekit.timed_objects import TimedText
    except ImportError as e:
        print(f"CRITICAL: Failed to import nllw or whisperlivekit: {e}")
        return

    # Mocks args
    src_lang = 'tel_Telu'
    tgt_lang = 'eng_Latn'
    
    print(f"Source Language: {src_lang}")
    print(f"Target Language: {tgt_lang}")

    # Initialize Model
    print("Loading model (600M)...")
    try:
        translation_params = { 
            "nllb_backend": "transformers",
            "nllb_size": "600M"
        }
        # Mocking args.lan list requirement if logic requires it
        translation_model = load_model([src_lang], **translation_params)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize Online Processor
    print("Initializing OnlineTranslation...")
    try:
        online_translator = OnlineTranslation(
            translation_model, 
            [src_lang], 
            [tgt_lang]
        )
        print("OnlineTranslation initialized.")
    except Exception as e:
        print(f"CRITICAL: Failed to init OnlineTranslation: {e}")
        return

    # Test Translation
    test_text = "నమస్కారం" # Hello in Telugu
    print(f"Testing translation for: '{test_text}'")
    
    # Create Token
    token = TimedText(text=test_text, start=0.0, end=1.0, detected_language=src_lang)
    
    # Insert
    print("Inserting token...")
    online_translator.insert_tokens([token])
    
    # Inspect Object
    print("Inspecting OnlineTranslation object...")
    print(f"DIR: {dir(online_translator)}")
    
    # Process
    print("Processing...")
    try:
        result, buffer = online_translator.process() # It returns tuple usually? In code it was used as such.
        # But wait, in my previous code I used: new_translation, new_translation_buffer = await asyncio.to_thread(self.translation.process)
        # So it returns TWO values?
        # The previous debug script did: result, _ = online_translator.process()
        # So that matches.
        
        print(f"RESULT TEXT: '{result.text}'")
        
        if not result.text:
             print("Result empty. Trying with PUNCTUATION...")
             token2 = TimedText(text=".", start=1.0, end=1.1, detected_language=src_lang)
             online_translator.insert_tokens([token2])
             res2, buf2 = online_translator.process()
             print(f"RESULT 2 TEXT: '{res2.text}'")
             res2, buf2 = online_translator.process()
             print(f"RESULT 2 TEXT: '{res2.text}'")
             
             # Attempt to use validate_buffer_and_reset
             print("Attempting validate_buffer_and_reset...")
             res3, buf3 = online_translator.validate_buffer_and_reset()
             print(f"RESULT 3 TEXT: '{res3.text}'")

    except Exception as e:
        print(f"Processing failed: {e}")
        
    print("-" * 20)
    print("Testing Underlying Model directly...")
    if hasattr(online_translator, 'translation_model'):
        tm = online_translator.translation_model
        print(f"Inner Model Type: {type(tm)}")
        print(f"Inner Model DIR: {dir(tm)}")
        
        # Try to find a translate method
        if hasattr(tm, 'translate_sentences'):
             print("Found translate_sentences method.")
    # Test 3+ words rule
    print("-" * 20)
    print("Testing 3+ Words Input (NLLW Limitation Check)...")
    # "Hello, how are you?" -> "నమస్కారం మీరు ఎలా ఉన్నారు" (4 words)
    long_text = "నమస్కారం మీరు ఎలా ఉన్నారు"
    token_long = TimedText(text=long_text, start=0.0, end=5.0, detected_language=src_lang)
    
    online_translator.insert_tokens([token_long])
    res_long, _ = online_translator.process()
    print(f"LONG INPUT ('{long_text}'): '{res_long.text}'")
    
    # Test Raw Model Access (Bypass)
    print("-" * 20)
    print("Testing Raw HF Model Access (to bypass <3 words limit)...")
    if hasattr(online_translator, 'translation_model'):
        tm = online_translator.translation_model
        raw_model = tm.translator # AutoModelForSeq2SeqLM
        tokenizer_dict = tm.tokenizer
        
        # Get tokenizer for source
        # tm.get_tokenizer expects NLLBI code, which converted_src_langs logic handles
        # We need to know what keys valid.
        # usually it is just the nllb code.
        
        tokenizer = tokenizer_dict.get(src_lang)
        if not tokenizer:
             print(f"Tokenizer for {src_lang} not in dict directly. Trying get_tokenizer...")
             tokenizer = tm.get_tokenizer(src_lang)
             
        if raw_model and tokenizer:
             print("Found raw model and tokenizer.")
             import torch
             
             short_text = "నమస్కారం"
             print(f"Attempting Raw Translation of '{short_text}'...")
             
             inputs = tokenizer(short_text, return_tensors="pt")
             # Move to device if needed
             device = tm.device
             inputs = {k: v.to(device) for k, v in inputs.items()}
             
             # We need to force target language.
             # NLLB uses forced_bos_token_id usually.
             
             # Need to find target lang id
             # From code: self.tokenizer.convert_tokens_to_ids(self.target_lang)
             # or lang_code_to_id
             
             forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
             print(f"Forced BOS ID for {tgt_lang}: {forced_bos_token_id}")
             
             try:
                 with torch.no_grad():
                     generated_tokens = raw_model.generate(
                         **inputs,
                         forced_bos_token_id=forced_bos_token_id,
                         max_length=50
                     )
                 raw_result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                 print(f"RAW RESULT: '{raw_result}'")
             except Exception as e:
                 print(f"Raw generation failed: {e}")
        else:
             print("Could not retrieve raw model or tokenizer.")
            


if __name__ == "__main__":
    test_nllb()

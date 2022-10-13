def check_prompts(model, prompts):
    max_length = 75 # The token buffer length is 77 but there's always a special start/end token which means only 75 are usable
    for p in prompts:
        tokenizer = model.cond_stage_model.tokenizer
        tokens = tokenizer.tokenize(p)
        ids    = tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) > max_length:
            overflow_ids = ids[max_length:]
            overflow_tokens = tokenizer.convert_ids_to_tokens(overflow_ids)
            overflow_text = ''.join(overflow_tokens)
            overflow_text = overflow_text.replace('</w>', ' ')
            print(f"WARNING: Your prompt is too long, the following text will be lost - '{overflow_text}'")

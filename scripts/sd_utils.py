from dataclasses import dataclass

@dataclass
class PromptInfo:
    prompt: str = ""
    neg_prompt: str = ""

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

def clean_prompt(prompt):
    result = prompt
    # get rid of any excessive spaces
    while '  ' in result:
        result = result.replace('  ', ' ')
    # get rid of whitespace at the start/end of the prompt
    return result.strip()

def read_prompts_from_file(filename):
    prompts_data = []
    with open(filename, "r") as f:
        prompts = f.read().splitlines()
        for prompt in prompts:
            pi = PromptInfo(prompt = clean_prompt(prompt))
            neg_pos = prompt.find('###')
            if neg_pos >= 0:
                pi.prompt = clean_prompt(prompt[0:neg_pos])
                pi.neg_prompt = clean_prompt(prompt[neg_pos+3:])
            # Only add non empty prompts and ignore any starting with #, those are considered commented out
            if (len(pi.prompt) > 0) and (pi.prompt[0] != '#'):
                prompts_data.append(pi)
    return prompts_data

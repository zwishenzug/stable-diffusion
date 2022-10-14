import numpy as np
import torch

class PromptInfo:
    def __init__(self, prompt, neg_prompt = ""):
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.buffs = []
        self.nerfs = []
        self.neg_buffs = []
        self.neg_nerfs = []
    def parse(self):
        self.prompt, self.buffs, self.nerfs = self.parse_prompt(self.prompt)
        self.neg_prompt, self.neg_buffs, self.neg_nerfs = self.parse_prompt(self.neg_prompt)
    def parse_prompt(self, prompt):
        # find nerfs and buffs
        pos = 0
        buffs = []
        nerfs = []
        remade_prompt = ''
        while pos < len(prompt):
            if prompt[pos] == '(':
                string = ''
                level = 0
                while (pos < len(prompt)) and (prompt[pos] == '('):
                    level += 1
                    pos += 1
                while (pos < len(prompt)) and (prompt[pos] != ')'):
                    string += prompt[pos]
                    pos += 1
                buffs.append([string, 1.1 ** level])
                remade_prompt += string
                while (pos < len(prompt)) and (prompt[pos] == ')'):
                    level -= 1
                    pos += 1
                assert level == 0, 'Mismatched ()'
            elif prompt[pos] == '[':
                string = ''
                level = 0
                while (pos < len(prompt)) and (prompt[pos] == '['):
                    level += 1
                    pos += 1
                while (pos < len(prompt)) and (prompt[pos] != ']'):
                    string += prompt[pos]
                    pos += 1
                nerfs.append([string, 1.1 ** level])
                remade_prompt += string
                while (pos < len(prompt)) and (prompt[pos] == ']'):
                    level -= 1
                    pos += 1
                assert level == 0, 'Mismatched []'
            else:
                remade_prompt += prompt[pos]
                pos += 1

        return remade_prompt, buffs, nerfs

def make_tokens(model, text, finalize = False):
    tokenizer = model.cond_stage_model.tokenizer
    tokens = tokenizer.tokenize(text)
    ids    = tokenizer.convert_tokens_to_ids(tokens)
    if finalize:
        ids = [tokenizer.bos_token_id] + ids[0:75]
        while len(ids) < 77:
            ids = ids + [tokenizer.eos_token_id]
    return ids

def check_prompt(model, p):
    max_length = 75 # The token buffer length is 77 but there's always a special start/end token which means only 75 are usable
    ids = make_tokens(model, p)
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

def buff_and_nerf_cond(model, cond, tokens, buffs, nerfs):
    assert(len(cond) == len(tokens))

    orig_mean = cond.mean()
    orig_std  = cond.std()

    for i in range(len(tokens)):
        for b in buffs:
            b_tokens = make_tokens(model, b[0])
            if np.array_equal(tokens[i:i+len(b_tokens)], b_tokens):
                for j in range(len(b_tokens)):
                    cond[i+j] *= b[1]
        for n in nerfs:
            n_tokens = make_tokens(model, n[0])
            if np.array_equal(tokens[i:i+len(n_tokens)], n_tokens):
                for j in range(len(n_tokens)):
                    cond[i+j] /= n[1]

    cond *= orig_std / cond.std()
    cond += orig_mean - cond.mean()

    return cond
    
def build_cond(model, device, batch_size, prompt, buffs, nerfs):
    tokens = make_tokens(model, prompt, True)
    check_prompt(model, prompt)
    cond = model.get_learned_conditioning([prompt])
    cond = buff_and_nerf_cond(model, cond[0].cpu().numpy(), tokens, buffs, nerfs)
    batch = []
    for _ in range(batch_size):
        batch.append(cond)
    return torch.tensor(np.array(batch)).to(device)

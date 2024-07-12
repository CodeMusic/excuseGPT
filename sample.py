import random
import sys 

prompt = "."
promptOutput = ""

import random
def get_emotion_tag():
    # random number with 5 posbilities which will map to ~NEUTRAL~, ~MAD~, ~SAD~, ~AFRAID~, or ~GLAD~
    random_number = random.randint(0, 4)
    emotion_tags = ['NEUTRAL', 'MAD', 'SAD', 'AFRAID', 'GLAD']
    return f"~{emotion_tags[random_number]}~"

def askCodeMusai(question = '.', system_message = '.', temperature = 0.9, useTokenEmbedding= True):
    global prompt, promptOutput
    prompt = question
    mood = get_emotion_tag()
    #if (prompt.startswith('.')):
    prompt = question + "\n" + mood + "\n" #force specific prompt
    promptOutput = f"Mood: {mood.replace('~', '').strip().title()}\nPrompt: {question}\n"
    return main(mood)

def main(inMood = ""):
    global prompt, promptOutput
    """
    Sample from a trained model
    """
    import os
    import pickle
    from contextlib import nullcontext
    import torch
    import tiktoken
    from model import GPTConfig, GPT

    # -----------------------------------------------------------------------------
    init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    out_dir = 'trainedModelOut' # ignored if init_from is not 'resume'
    useChar = False
    if (useChar):
        out_dir = f'{out_dir}_char'
    else:
        out_dir = f'{out_dir}_token'

    #if prompt.startswith('.'):
    #    prompt = get_emotion_tag()
    #elif (inMood != ""):
    prompt = inMood

    start = prompt # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 1 # number of samples to draw
    max_new_tokens = 100 # number of tokens generated in each sample
    temperature = 1.19 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
    #seed = 1337
    device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
 
    
    print(f"start: {start}")
    print(f"start_ids: {start_ids}") 

    stopIdx = encode('~')[0]
    print('\n---------------\n')

    temperature = 0.05
    tempIncrement = 0.05
    incrementTemp = True
    randomTemp = True
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):

                if (randomTemp):
                    temperature = random.uniform(0.01, 0.9)
                    if (random.randint(0, 100) == 50):
                        temperature = 3
                    elif (random.randint(0, 100) < 50):
                        temperature = random.uniform(0.01, 0.9)
                    else:
                        temperature = random.uniform(1.0, 2.0)
                
                if (incrementTemp):
                    temperature += tempIncrement
                    if (temperature > 2.48):
                        tempIncrement = -0.05
                    elif (temperature < 0.05):
                        tempIncrement = 0.05


                #temperature = round(temperature, 2)

                print(f"{promptOutput}")
                print("-----------------")
                print(f"{prompt}")
                y = model.generate(x, max_new_tokens, decode=decode, temperature=temperature, top_k=top_k, stopIdx=stopIdx)
                out = decode(y[0].tolist()).replace('\n', '').replace('.', '') + '.'
                thoughtVariabilityText = f"({temperature:.2f})"
                print(f"                   {thoughtVariabilityText}")
                #print(out.strip())
                #print('\n')
    
    return f"{inMood}\n\n{out}\n\n{thoughtVariabilityText}"


if __name__ == "__main__":
    #the arg I entered in the conmmand line after the file name did not get passed as an argument, why?
    #I want to be able to pass in a mood as a command line argument and have it passed to the main function
    #I can do this by adding a new argument to the command line
    #I can do this by adding a new argument to the command line

    args = sys.argv[1:]
    if (len(args) > 0):
        print(f"args: {args[0]}")
        main("~" + args[0] + "~")
    else:
        main()
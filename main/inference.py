import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from main.architecture import Model 
from main.config import Config


@torch.no_grad
def sampler(logits,p=0.9,temp=1) -> torch.tensor:    # nucleus(top-p) sampling
    logits = logits / temp
    probs = F.softmax(logits,dim=-1)
    #sort probs
    sorted_probs,sorted_ids = torch.sort(probs,descending=True)
    #mask
    cum_probs = torch.cumsum(sorted_probs,dim=-1)
    mask = cum_probs > p
    #fillter
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0
    ids_to_remove = sorted_ids[mask]
    #fillter logits
    filt_logits = logits.clone()
    filt_logits[:,ids_to_remove] = -float('Inf')

    probs = F.softmax(filt_logits,dim=-1)
    next_tok = torch.multinomial(probs,num_samples=1)
    return next_tok

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("main/checkpoint/Onyx(1).pt", map_location=device)
model = Model(Config)
model.load_state_dict(checkpoint["model_state"], strict=True)
tok = ByteLevelBPETokenizer(
    "main/tokenizer/vocab-o.json",
    "main/tokenizer/merges-o.txt",
        )


@torch.no_grad
def gen(prompt, max_tok=900, temperature=0.7, eos_token="<|EOS|>"):
    # Ensure tokenizer is properly configured for special tokens if they are added
    # The format function in the training loop suggests <bos> and <eos> are used.
    model.eval()
    eos_token_id = tok.token_to_id(eos_token) if tok.token_to_id(eos_token) is not None else -1

    inp_ids = tok.encode(prompt)
    ids = torch.tensor([inp_ids.ids], dtype=torch.long).to(device)
    cur_pos = len(inp_ids)
    for _ in range(max_tok):    #token stream
        logits = model(ids)[:, -1, :]
        next_id = sampler(logits)
        if next_id.item() == eos_token_id:
            break

        ids = torch.cat([ids, next_id], dim=-1)

        new_text = tok.decode(ids[0,cur_pos:].cpu().tolist())
        if new_text:
            yield new_text
            cur_pos = ids.shape[1]






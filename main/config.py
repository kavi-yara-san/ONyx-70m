from dataclasses import dataclass
import torch 

@dataclass
class Config:
    
    """
    Attributes:
        dim : int - Model embedding dimension. Must be divisible by heads.
        num_exp : int - Number of router experts in MoE.
        top_k : int - top k experts per token
        heads : int - attention head for mha
        vocab_size : int - vocab size used in tokenizer
        max_seq_len : int - context window
        device : torch.device [CUDA / CPU]
                   

    """

    dim: int = 512

    #experts config moe
    num_exp: int = 4
    num_shared_exp: int = 1
    top_k: int = 1

    # mha
    heads: int = 8
    vocab_size: int = 8000
    max_seq_len: int = 1024

    #train
    batch_size: int = 32
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

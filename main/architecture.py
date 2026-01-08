import torch 
import torch.nn as nn
from main.layers.embeddings import RoPE
from main.layers.transformer import Transformer_block
from main.config import Config

class Model(nn.Module):
    def __init__(self,param:Config):
        """
        param : Config
          Global configuration object defining model dimensions,
        attention heads, vocabulary size, and sequence length.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(param.vocab_size,param.dim)
        self.block = nn.ModuleList([Transformer_block(param) for _ in range(10)])
        self.lin = nn.Linear(param.dim, param.vocab_size,bias=False)
        self.lin.weight.data = self.tok_emb.weight.data.clone()
        self.rope_emb = RoPE(param)
        self.norm = nn.RMSNorm(param.dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self,module)->None:  # weight intitailization
      if isinstance(module,nn.Embedding):
        torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

      if isinstance(module,nn.Linear):
        if hasattr(module,'is_residual') and module.is_residual:
          torch.nn.init.normal_(module.weight,mean=0.0,std=0.03)
        else:
          torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)

    def forward(self,x) -> torch.tensor:
        """Args:
              x : torch.tensor 
           Returns:
              x : torch.tensor
          """
        
        x = self.tok_emb(x)
        x = self.rope_emb(x)
        for block in self.block:
            x = block(x)
        h = self.norm(x)
        x = self.lin(h)
        return x
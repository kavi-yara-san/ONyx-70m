import torch
import torch.nn as nn
from main.config import Config
class RoPE(nn.Module):
    def __init__(self,config:Config):
        """
        Args :
            config : Config
            Global configuration providing embedding dimension,
            maximum sequence length, and device placement.
        
        """
        super().__init__()
        self.dim = config.dim
        self.max_seq_len = config.max_seq_len
        self.device = config.device
    def compute_theta(self,seq_len) -> torch.tensor:
        base = 10000**(-torch.arange(0,self.dim,2).float()/self.dim)
        pos = torch.arange(seq_len).unsqueeze(dim=1)
        theta = base*pos
        return theta.to(self.device)
    def apply_rope(self,x,theta) -> torch.tensor:

        half_dim = self.dim//2
        q1 = x[...,half_dim:]
        q2 = x[...,:half_dim]
        x1 = (q1*torch.cos(theta).unsqueeze(0)) - (q2*torch.sin(theta).unsqueeze(0))
        x2 = (q1*torch.sin(theta).unsqueeze(0)) + (q2*torch.cos(theta).unsqueeze(0))
        rope = torch.cat([x1,x2],dim=-1)
        return rope.view(x.shape)

    def forward(self,x)-> torch.tensor:
        """
        Forward pass applying RoPE.

        Args :
        
        x : torch.Tensor
            Input embeddings of shape(batch, seq_len, dim)

        Returns :
       
            torch.Tensor
                Position-aware embeddings of
                shape(batch, seq_len, dim)
        """

        bt,seq,dim = x.shape
        assert seq<=self.max_seq_len,'seq len must into '
        theta = self.compute_theta(seq)
        x_out = self.apply_rope(x,theta)

        return x_out

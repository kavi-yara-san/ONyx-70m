import torch
import torch.nn as nn 
from main.layers.attention import Mha
from main.layers.MoEmodule import MoEModule
from main.config import Config


class Transformer_block(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.dim = config.dim
        self.heads = config.heads
        self.multiheadattention = Mha(dim=self.dim,head=self.heads)
        self.norm1 =  nn.RMSNorm(self.dim, eps=1e-6)
        self.moe = MoEModule(config)
        self.norm2 = nn.RMSNorm(self.dim, eps=1e-6)


    def forward(self,xi) -> torch.tensor:

        x = xi
        b,seq_len,dim = x.shape
        x_out = x + self.multiheadattention(self.norm1(x)) # pre_norm -> mha & resiudal
        xmoe = x_out + 0.1 * self.moe(self.norm2(x_out))

        return xmoe
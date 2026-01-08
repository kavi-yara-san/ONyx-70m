import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
class Mha(nn.Module):
    def __init__(self,dim,head):
        """
        Args : 
            dim : torch.tensor 
            head : int [attention head]

        """

        super().__init__()
        self.dim = dim
        self.head = head
        assert self.dim % self.head == 0,'dim must be divisible by head'
        self.dk = self.dim // self.head

        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.qw = nn.Linear(dim, dim, bias=False)


        self.linproj = nn.Linear(self.dim,self.dim)

    def forward(self,x) -> torch.tensor:
    

        b,s,d = x.shape
        k = self.kw(x)
        v = self.vw(x)
        q = self.qw(x)
        #split the q,k,v
        k = k.view(b, s, self.head, self.dk).transpose(1, 2)
        v = v.view(b, s, self.head, self.dk).transpose(1, 2)
        q = q.view(b, s, self.head, self.dk).transpose(1, 2)
        #compute attn
        mask = torch.tril(torch.ones(s, s, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(1).expand(b, self.head, s, s)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)    # (b, h, s, s)
        attn = attn.masked_fill(~mask,-1e4)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v           # (b, h, s, dk)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        #linear projection
        out = self.linproj(out)
        return out

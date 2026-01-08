
import torch 
import torch.nn as nn
import torch.nn.functional as F
from main.config import Config

class Expert(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc = nn.Sequential(

            nn.Linear(dim,dim*2),
            nn.GELU(),
            nn.Linear(dim*2,dim)


        )
    def forward(self,x)-> torch.tensor:
        x = self.fc(x)
        return x


class SharedExpert(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc = nn.Sequential(

            nn.Linear(dim,dim*2),
            nn.GELU(),
            nn.Linear(dim*2,dim)


        )
    def forward(self,x)-> torch.tensor:
        x = self.fc(x)
        return x

class Gating(nn.Module):
    def __init__(self,dim,num_exp,top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Sequential(

            nn.Linear(dim,dim*2),
            nn.GELU(),
            nn.Linear(dim*2,num_exp)
        )

    def forward(self,x) -> torch.tensor:
        route = self.gate(x)
        scores = F.softmax(route,dim=-1)
        #selection of top k
        weights,indices = torch.topk(scores,self.top_k,dim=-1)
        weights = weights / weights.sum(dim=-1,keepdim=True)
        return weights,indices
    


class MoEModule(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.dim = config.dim
        self.num_router_exp = config.num_exp
        self.num_shared_exp = config.num_shared_exp
        self.top_k = config.top_k
        self.gating = Gating(self.dim,self.num_router_exp,self.top_k)
        self.task_spc_experts = nn.ModuleList([Expert(self.dim) for _ in range(self.num_router_exp)])
        self.shared_experts = nn.ModuleList([SharedExpert(self.dim) for _ in range(self.num_shared_exp)])

    def forward(self,x) -> torch.tensor:
        batch_size,seq_len,dim = x.shape
        x_flatten = x.view(-1,dim)
        weights,indices = self.gating(x_flatten)
        #routed experts process
        shared_experts_output = torch.zeros_like(x_flatten)
        task_output = torch.zeros_like(x_flatten)

        for i in range(self.num_shared_exp):
            shared_experts_output += self.shared_experts[i](x_flatten)

        for i in range(self.top_k):
            idx = indices[:,i]
            weight = weights[:,i]
            for j in range(self.num_router_exp):
                mask = (idx==j)
                experts_input = x_flatten[mask]
                experts_output = self.task_spc_experts[j](experts_input)
                task_output[mask] += experts_output*weight[mask].unsqueeze(-1)

        output = task_output + shared_experts_output


        return output.view(batch_size,seq_len,dim)

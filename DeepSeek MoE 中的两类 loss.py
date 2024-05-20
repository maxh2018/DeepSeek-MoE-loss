import torch ,math
import torch.nn.functional as F
import torch.nn as nn




class MoEGate(nn.Module):
    def __init__(self):#, config
        super().__init__()
        #添加的额外参数
        self.CommBal=True
        self.n_routed_experts=8
        self.alpha2 = 0.001         #config.aux_loss_alpha
        self.alpha3 = 0.001         #config.aux_loss_alpha
        self.training=True
        self.DevBal=True
        self.CommBal=True
        self.M=1
        self.D=4

        #self.config = config
        self.top_k = 2 #config.num_experts_per_tok
        #self.n_routed_experts = config.n_routed_experts

        self.scoring_func ="softmax"   #config.scoring_func
        self.alpha =0.001          #config.aux_loss_alpha
        self.seq_aux =False #config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob =False #config.norm_topk_prob
        self.gating_dim =12 #config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #print("self.weight=",self.weight)
    def forward(self, hidden_states):
            bsz, seq_len, h = hidden_states.shape        
            ### compute gating score
            hidden_states = hidden_states.view(-1, h)
            logits = F.linear(hidden_states, self.weight, None)
            if self.scoring_func == 'softmax':
                scores = logits.softmax(dim=-1)
            else:
                raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
            
            ### select top-k experts
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
            
            ### norm gate to sum 1
            if self.top_k > 1 and self.norm_topk_prob:
                denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                topk_weight = topk_weight / denominator
            #print("topk_idx=\n",topk_idx,"\n")
            ### expert-level computation auxiliary loss
            if self.training and self.alpha > 0.0:
                scores_for_aux = scores  #(bsz*seq_len,n_routed_experts)
                aux_topk = self.top_k
                # always compute aux loss based on the naive greedy topk method
                topk_idx_for_aux_loss = topk_idx.view(bsz, -1) #(bsz,seq_len*k)
                if self.seq_aux:
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                    ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                    ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                    aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
                if self.DevBal:
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) #(bsz,seq_len,n_routed_experts)
                    #print("scores_for_seq_aux=\n",scores_for_seq_aux,"\n")
                    Pi_tensor=scores_for_seq_aux.mean(dim = 1) #(bsz,self.n_routed_experts)
                    #print("Pi_tensor=\n",Pi_tensor,"\n")
                    ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                    ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                    fi_tensor=ce  #(bsz,self.n_routed_experts)
                    #print(f"fi_tensor=({fi_tensor.size()})\n",fi_tensor,"\n")
                    
                    D_group_expert_list=[]#list(range(self.n_routed_experts))
                    P_i_list=[]
                    D=4
                    assert self.n_routed_experts%D==0
                    step=int(self.n_routed_experts/D)
                    for i in range(0,self.n_routed_experts,step):
                        D_group_expert_list.append(fi_tensor[:,i:i+step].mean(dim=-1,keepdim=True)) ######
                        P_i_list.append(Pi_tensor[:,i:i+step].sum(dim=-1,keepdim=True))
                    fi__total_tensor,P_i_total_tensor=None,None    
                    for ele in D_group_expert_list:
                        if fi__total_tensor is None:
                            fi__total_tensor=ele
                        else:
                            fi__total_tensor=torch.cat((fi__total_tensor,ele),dim=-1)  #(bsz,D)
                    #print("程序输出的fi__total_tensor=\n",fi__total_tensor)
                    for ele in P_i_list:
                        if  P_i_total_tensor is None:
                            P_i_total_tensor=ele
                        else:
                            P_i_total_tensor=torch.cat((P_i_total_tensor,ele),dim=-1) #(bsz,D)
                    #print("程序输出的P_i_total_tensor=\n",P_i_total_tensor)
                    #print("fi__total_tensor*P_i_total_tensor=\n",fi__total_tensor*P_i_total_tensor)
                    DevBal_loss=(fi__total_tensor*P_i_total_tensor).sum(dim=1).mean()* self.alpha2
                if self.CommBal:
                    scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1) #(bsz, seq_len, n_routed_experts)
                    #print("scores_for_seq_aux=\n",scores_for_seq_aux,"\n")
                    raw=topk_idx.view(bsz,seq_len, -1) #(bsz, seq_len, k)
                    #print("topk_idx.view=\n",raw,"\n")
                    assert self.n_routed_experts%D==0
                    step=int(self.n_routed_experts/D)
                    #print("before raw=\n",raw)
                    raw=raw//step
                    #print("raw=\n",raw)
                    new=torch.zeros(bsz,D)
                    for i in range(D):
                        tem= (raw==i).any(dim=-1).sum(dim=-1)
                        new[:,i]=tem
                    #print("new=",new)
                    new=new*D/(seq_len*self.M)
                    Pi_tensor=scores_for_seq_aux.mean(dim = 1) #(bsz,self.n_routed_experts)
                    P_i_list=[]
                    D=4
                    for i in range(0,self.n_routed_experts,step):
                        P_i_list.append(Pi_tensor[:,i:i+step].sum(dim=-1,keepdim=True))
                    P_i_total_tensor=None
                    for ele in P_i_list:
                        if  P_i_total_tensor is None:
                            P_i_total_tensor=ele
                        else:
                            P_i_total_tensor=torch.cat((P_i_total_tensor,ele),dim=-1)
                    CommBal_loss=(new*P_i_total_tensor).sum(dim=1).mean()* self.alpha3               
            #else:
            #    aux_loss,DevBal_loss = None,None
            return topk_idx, topk_weight,DevBal_loss,CommBal_loss  # aux_loss,


MOE=MoEGate()
bsz, seq_len, h=(2,3,12)
hidden_states=torch.rand((bsz, seq_len, h))
MOE(hidden_states)
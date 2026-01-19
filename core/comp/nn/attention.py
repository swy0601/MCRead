import math
from typing import Optional

import torch

from allennlp.common.registrable import Registrable
import torch.nn.functional as F
from sklearn.cluster import KMeans
class SeqAttention(Registrable, torch.nn.Module):
    def __init__(self):
        super(SeqAttention, self).__init__()

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor,
                index:torch.Tensor,
                typeAdd_Del:bool):
        """
        @param mask: Note this mask is k/v mask, commonly pad_mask.
        """
        raise NotImplementedError

    def attend_by_scores(self, v_exp, scores, mask_exp):
        """
        Giving attention scores, expanded values and expanded mask,
        this method will mask scores and then using softmax to produce
        probabilities as weight to sum values.
        """
        # shape: [batch, qlen, klen]
        # Apply mask, note used mask is the 'not' result of the original mask
        scores = scores.masked_fill(~mask_exp, -float('inf'))

        # compute probs
        probs = torch.softmax(scores, dim=-1).unsqueeze(-1)

        attended = (v_exp * probs).sum(dim=2)

        return attended


@SeqAttention.register('mlp')
class MlpSeqAttention(SeqAttention):
    def __init__(self, k_dim, q_dim, **kwargs):
        super(MlpSeqAttention, self).__init__()

        self.linear_fusion_q = torch.nn.Linear(q_dim, k_dim, bias=False)
        self.linear_fusion_k = torch.nn.Linear(k_dim, k_dim, bias=False)
        self.linear_fusion_v = torch.nn.Linear(k_dim, 1, bias=False)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor):
        """
        Use query to attend add and del code features.
        @param q: Shape: [batch, dim] or [batch, seq, dim]
        @param k: Shape: [batch, seq, dim]
        @param v: Shape: [batch, seq, dim]
        @param mask: mask for key/value sequence
        """
        # adapt for 2D input, without seq dim
        expand_q = len(q.shape) == 2
        if expand_q:
            q = q.unsqueeze(1) # .repeat((1, dim_1, 1))

        assert k.size(1) == v.size(1) == mask.size(1)
        q_seq_len, k_seq_len = q.size(1), k.size(1)
        # shape: [batch, qlen, klen, dim]
        q_exp = q.unsqueeze(2).repeat((1,1,k_seq_len,1))
        k_exp = k.unsqueeze(1).repeat((1,q_seq_len,1,1))
        v_exp = v.unsqueeze(1).repeat((1,q_seq_len,1,1))
        mask_exp = mask.unsqueeze(1).repeat((1,q_seq_len,1))

        scores = torch.tanh(self.linear_fusion_q(q_exp) + self.linear_fusion_k(k_exp))
        # scores shape: [batch, seq]
        scores = self.linear_fusion_v(scores).squeeze(-1)

        attended = self.attend_by_scores(v_exp, scores, mask_exp)
        if expand_q:
            attended = attended.squeeze(1)

        return attended


@SeqAttention.register('attention')
class Attention(SeqAttention):
    def __init__(self,
                 input_size: int,
                 k_dim: Optional[int] = None,
                 v_dim: Optional[int] = None,
                 dropout: float = 0.5,
                 head_nums: int = 1,
                 **kwargs):
        super(Attention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=input_size,
                                                     num_heads=head_nums,
                                                     dropout=dropout,
                                                     kdim=k_dim,
                                                     vdim=v_dim)

        self.fc = torch.nn.Linear(input_size, input_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(input_size)
        self.attn_gate = torch.nn.Sequential(
            torch.nn.Linear(768, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
            torch.nn.Sigmoid()
        )

    def get_neg_bias(self,neg_bias_raw):
        return -F.softplus(neg_bias_raw)





    def glbal_mask(self,logits,q,k,v,self_topk_rate,self_enlarge):
        cross_max = logits.amax(dim=(1,2), keepdim=True)
        d = q.shape[-1]
        raw_self_scores = torch.einsum("qbd,kbd->bqk",k,k) / math.sqrt(k.size(-1))
        w = raw_self_scores.sum(dim=-2)  # [B, k]
        topk = max(1, int(w.size(-1) * self_topk_rate))
        topk_vals, topk_idx = torch.topk(w, k=topk, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(1, topk_idx, 1.0)

        cross_max_expanded = (
            cross_max.squeeze(-1)
                      .squeeze(-1)
                      .unsqueeze(-1)
                      .expand_as(w)
        )

        w_mapped = -cross_max_expanded / 3.0
        w_mapped = torch.where(mask.bool(), cross_max_expanded, w_mapped)
        w_mapped = w_mapped.unsqueeze(1)
        return w_mapped

    def forward(self, q, k, v, mask,index,typeAdd_Del):
        expand_seq_dim = len(q.shape) == 2
        if expand_seq_dim:
            q = q.unsqueeze(1)

        # reshape to [seq, batch, dim]
        q = q.transpose(0,1).contiguous()
        k = k.transpose(0,1).contiguous()
        v = v.transpose(0,1).contiguous()

        mask_fixed = mask.clone()
        valid_rows = ( mask_fixed == False).any(dim=1)
        if not valid_rows.all():
            for i, valid in enumerate(valid_rows):
                if  valid:
                    first_false_pos = 0
                    mask_fixed[i, first_false_pos] = True


        d = q.shape[-1]
        logits = torch.einsum("qbd,kbd->bqk", q, k) / math.sqrt(d)
        

        self_mask = self.glbal_mask(logits,q,k,v,self_topk_rate = 0.05,self_enlarge = 3)
 


        device = logits.device
        self_mask = self_mask.to(device)
 

        final_weights = logits + self_mask 

        final_weights = F.softmax(final_weights, dim=-1)

        v = v.transpose(0,1)
        attn_output = torch.einsum("bqs,bkd->bqd", final_weights, v)
        attn_output = attn_output.transpose(0,1).contiguous()
        attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1e5, neginf=-1e5)
        attn_output = self.dropout(self.fc(attn_output))
        attn_output = self.layer_norm(attn_output + q).transpose(0,1).contiguous()
        if expand_seq_dim:
            return attn_output.squeeze(1)
        else:
            return attn_output


@SeqAttention.register('simple_multi_head')
class SimpleMultiHeadSeqAttention(SeqAttention):
    """
    Using pytorch multi-head attention to attend, but without fc, dropout,
    layer norm and residual connection.
    """
    def __init__(self,
                 input_size: int,
                 dropout: float = 0.5,
                 head_nums: int = 1,
                 **kwargs):
        super(SimpleMultiHeadSeqAttention, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=input_size,
                                                     num_heads=head_nums,
                                                     dropout=dropout)


    def forward(self, q, k, v, mask):
        expand_seq_dim = len(q.shape) == 2
        if expand_seq_dim:
            q = q.unsqueeze(1)

        # reshape to [seq, batch, dim]
        q = q.transpose(0,1).contiguous()
        k = k.transpose(0,1).contiguous()
        v = v.transpose(0,1).contiguous()

        # pytorch use the opposite mask as we have
        attn_output, _weights = self.attention.forward(q, k, v, key_padding_mask=~mask)
        attn_output = attn_output.transpose(0,1).contiguous()

        # squeeze seq dim if expanded
        if expand_seq_dim:
            return attn_output.squeeze(1)
        else:
            return attn_output


@SeqAttention.register('raw_return')
class RawReturnSeqAttention(SeqAttention):
    """
    Directly return q, k or v instead of making attention.
    """
    def __init__(self, return_name: str = 'q'):
        super(RawReturnSeqAttention, self).__init__()
        if return_name == 'q':
            self.return_func = lambda q,k,v: q
        elif return_name == 'k':
            self.return_func = lambda q,k,v: k
        elif return_name == 'v':
            self.return_func = lambda q,k,v: v
        else:
            raise ValueError


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor):
        return self.return_func(q, k, v)


@SeqAttention.register('dot_product')
class DotProductAttention(SeqAttention):
    """
    Very similar to implementation of pytorch MultiHeadTransformer, but always
    set num_head=1 and allow disable projection of qkv and output.
    Also, scaled dot-product is optional.
    """
    def __init__(self,
                 use_projection: bool,
                 q_input_dim: int,
                 k_input_dim: int,
                 v_input_dim: int,
                 proj_dim: int = 512,
                 scaled_dot_product: bool = True):
        super(DotProductAttention, self).__init__()

        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.scaled_dot_product = scaled_dot_product
        if use_projection:
            self.proj_q = torch.nn.Linear(q_input_dim, proj_dim, bias=False)
            self.proj_k = torch.nn.Linear(k_input_dim, proj_dim, bias=False)
            self.proj_v = torch.nn.Linear(v_input_dim, proj_dim, bias=False)
            self.proj_out = torch.nn.Linear(proj_dim, v_input_dim, bias=False)


    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor):

        expanded_q = len(q.shape) == 2
        if expanded_q:
            q = q.unsqueeze(1) # .repeat((1, dim_1, 1))

        if self.use_projection:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)
            feature_dim = self.proj_dim
        else:
            feature_dim = q.size(-1)

        ################################################
        # mask_exp shape: [batch, q_len, k_len]
        # v_exp shape: [batch, q_len, k_len, dim]
        # scores shape: [batch, q_len, k_len],
        ################################################
        assert k.size(1) == v.size(1) == mask.size(1)
        q_seq_len, k_seq_len = q.size(1), k.size(1)
        mask_exp = mask.unsqueeze(1).repeat((1,q_seq_len,1))
        v_exp = v.unsqueeze(1).repeat((1, q_seq_len, 1, 1))

        # use bmm to get free from repeat
        scores = torch.bmm(q, k.transpose(1,2))
        if self.scaled_dot_product:
            scores /= (feature_dim ** 0.5)

        attended = self.attend_by_scores(v_exp, scores, mask_exp)
        if expanded_q:
            attended = attended.squeeze(1)

        if self.use_projection:
            return self.proj_out(attended)
        else:
            return attended



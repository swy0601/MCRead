from typing import Dict, Optional
import torch
import os
import math
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric
from overrides import overrides

from core.comp.nn.fusion.sivo_fusions.sivo_fusion import SeqinVecoutFusion
from core.comp.nn.classifier import Classifier
from core.comp.nn.loss_func import LossFunc
from utils.metric import update_metric

# 20251022 wjk>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import torch
import torch.nn as nn
from einops import rearrange
from math import sqrt


class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, kernel=[3, 5, 7], stride=[1, 1, 1], padding=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        # 可学习 TopK 比率参数
        self.k_ratio1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.k_ratio2 = nn.Parameter(torch.tensor(0.25), requires_grad=True)

        # 1D 可变卷积代替池化（按序列长度维度卷积）
        # 输入shape: (B, dim, seq_len)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel[0], stride=stride[0], padding=padding[0], groups=dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel[1], stride=stride[1], padding=padding[1], groups=dim)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=kernel[2], stride=stride[2], padding=padding[2], groups=dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        x: Tensor of shape (B, seq_len_x, dim) - query input
        y: Tensor of shape (B, seq_len_y, dim) - key/value input feature map
        """

        B, N_y, C = y.shape
        B, N_x, _ = x.shape

        # multi-scale convolution on y
        # y -> (B, dim, seq_len)
        y_perm = y.permute(0, 2, 1)  # (B, C, N_y)

        y1 = self.conv1(y_perm)
        y2 = self.conv2(y_perm)
        y3 = self.conv3(y_perm)

        # 由于stride=1，padding设计，输出长度一般不同，统一长度到y1长度
        target_len = y1.shape[-1]
        y2 = nn.functional.interpolate(y2, size=target_len, mode='linear', align_corners=False)
        y3 = nn.functional.interpolate(y3, size=target_len, mode='linear', align_corners=False)

        y_sum = y1 + y2 + y3  # (B, C, target_len)
        y_sum = y_sum.permute(0, 2, 1)  # (B, target_len, C)

        y_norm = self.layer_norm(y_sum)  # LayerNorm applies on last dim

        # 计算 q, k, v
        kv = self.kv(y_norm).reshape(B, target_len, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, target_len, head_dim)

        q = self.q(x).reshape(B, N_x, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                   3)  # (B, num_heads, N_x, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_x, target_len)

        # 动态 TopK
        k1_num = (target_len * self.sigmoid(self.k_ratio1)).int().clamp(1, target_len)
        k2_num = (target_len * self.sigmoid(self.k_ratio2)).int().clamp(1, target_len)

        def topk_mask(attn, k_num):
            mask = torch.zeros_like(attn)
            index = torch.topk(attn, k=k_num.item(), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn_masked = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            attn_masked = attn_masked.softmax(dim=-1)
            attn_masked = self.attn_drop(attn_masked)
            return attn_masked

        attn1 = topk_mask(attn, k1_num)
        attn2 = topk_mask(attn, k2_num)

        out1 = attn1 @ v  # (B, num_heads, N_x, head_dim)
        out2 = attn2 @ v

        # 加权融合
        out = 0.6 * out1 + 0.4 * out2

        out = out.transpose(1, 2).reshape(B, N_x, C)  # (B, N_x, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        # 残差连接
        out = out + x

        return out


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


@Model.register('hybrid_imp_seqin_classifier')
class ImpClassifier(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            code_embedder: TextFieldEmbedder,
            token_code_embedder:TextFieldEmbedder,
            # [Note] This classifier assumes code feature has been collapsed into a single vector representation before fusion
            code_encoder: Seq2SeqEncoder,
            fusion: SeqinVecoutFusion,
            classifier: Classifier,
            loss_func: LossFunc,
            msg_embedder: Optional[TextFieldEmbedder] = None,
            msg_encoder: Optional[Seq2VecEncoder] = None,
            op_embedder: TextFieldEmbedder = None,
            metric: Metric = None,
            # initializer: InitializerApplicator = InitializerApplicator(),
            **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self._code_embedder = code_embedder
        self.token_code_embedder = token_code_embedder
        self._code_encoder = code_encoder
        self._msg_embedder = msg_embedder
        self._msg_encoder = msg_encoder
        self._op_embedder = op_embedder
        self._fusion = fusion
        self._classifier = classifier
        self._loss_func = loss_func
        self._metric = metric
        self.debug_forward_count = 0
        # 20251022 wjk>>>>>>>>>>>>>>>>>>>>>>>>
        self.model = MSC(dim=768, num_heads=8).cuda()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self._batch_counter = 0  # 👈 初始化计数器
        self._batch_counter1 = 0

        # =================== Attention Pooling 权重层 ===================
        D = self._code_embedder.get_output_dim() if hasattr(self._code_embedder, 'get_output_dim') else 768
        self.patch_attn_W = nn.Linear(D, D)
        self.patch_attn_v = nn.Linear(D, 1, bias=False)

        # =================== BiLSTM聚合方式 ===================
        self.patch_lstm_hidden = D // 2  # 双向拼接后维度为 D
        self.patch_bilstm = torch.nn.LSTM(
            input_size=D,
            hidden_size=self.patch_lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # 如果想用任意 hidden_size，也可以加 linear 映射回 D
        self.patch_bilstm_fc = torch.nn.Linear(2 * self.patch_lstm_hidden, D)

        # =================== Gated Pooling 动态融合 ===================

        # Gated Pooling 权重
        self.patch_gate_W = nn.Linear(D, D)

        self.patch_gate_W = nn.Linear(D, D)  # 可学习线性层
        # ---------------
        self.patch_W = 15
        self.patch_conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=768,
            kernel_size=(3, 3),
            stride=1,
            padding=(3 // 2, 3 // 2)
        )
        self.patch_pool2d = nn.AdaptiveMaxPool2d((1, 1))
        # initializer(self)

    def forward_features(self,
                         diff_add: TextFieldTensors,
                         diff_del: TextFieldTensors,
                         diff_op: TextFieldTensors = None,
                         msg: TextFieldTensors = None,
                         additional_feature: Optional[torch.Tensor] = None,
                         add_op_mask: Optional[torch.Tensor] = None,
                         del_op_mask: Optional[torch.Tensor] = None,
                         add_line_idx: Optional[torch.Tensor] = None,
                         del_line_idx: Optional[torch.Tensor] = None,
                         **kwargs) -> torch.Tensor:
        """
        Forward input features and output extracted cc features.
        """
        #
        # add_code_feature = self._encode_text_field(diff_add, self._code_embedder, self._code_encoder)
        # del_code_feature = self._encode_text_field(diff_del, self._code_embedder, self._code_encoder)

        add_code_feature = self._encode_text_field(diff_add, self.token_code_embedder, self._code_encoder)
        del_code_feature = self._encode_text_field(diff_del, self.token_code_embedder, self._code_encoder)

        if diff_op is not None:
            op_features = self._op_embedder(diff_op)
        else:
            op_features = None

        cc_feature = self._fusion(add_code_feature, del_code_feature, op_features,
                                  add_op_mask, del_op_mask,
                                  add_line_idx, del_line_idx)
        cc_repre = cc_feature['encoder_outputs']

        if msg is not None:
            msg_feature = self._encode_text_field(msg, self._msg_embedder, self._msg_encoder)
            msg_repre = msg_feature['encoder_outputs']
            cc_repre = torch.cat((cc_repre, msg_repre), dim=-1)

        if additional_feature is not None:
            cc_repre = torch.cat((cc_repre, additional_feature), dim=-1)

        return cc_repre

    def forward_features1(self,
                          diff_add: TextFieldTensors,
                          diff_del: TextFieldTensors,
                          diff_op: TextFieldTensors = None,
                          msg: TextFieldTensors = None,
                          additional_feature: Optional[torch.Tensor] = None,
                          add_op_mask: Optional[torch.Tensor] = None,
                          del_op_mask: Optional[torch.Tensor] = None,
                          add_line_idx: Optional[torch.Tensor] = None,
                          del_line_idx: Optional[torch.Tensor] = None,
                          **kwargs) -> torch.Tensor:
        """
        Forward input features and output extracted cc features.
        """
        # 利用cannie进行嵌入！！！（正确的）
        add_code_feature,add_op_mask = self.embed_diff_add2(diff_add,add_op_mask)
        del_code_feature,del_op_mask = self.embed_diff_add2(diff_del,del_op_mask)

        # 使用卷积操作
        # add_code_feature, add_op_mask = self.embed_diff_add3(diff_add, add_op_mask)
        # del_code_feature, del_op_mask = self.embed_diff_add3(diff_del, del_op_mask)

        if diff_op is not None:
            op_features = self._op_embedder(diff_op)
        else:
            op_features = None

        cc_feature = self._fusion(add_code_feature, del_code_feature, op_features,
                                  add_op_mask, del_op_mask,
                                  add_line_idx, del_line_idx)
        cc_repre = cc_feature['encoder_outputs']

        if msg is not None:
            msg_feature = self._encode_text_field(msg, self._msg_embedder, self._msg_encoder)
            msg_repre = msg_feature['encoder_outputs']
            cc_repre = torch.cat((cc_repre, msg_repre), dim=-1)

        if additional_feature is not None:
            cc_repre = torch.cat((cc_repre, additional_feature), dim=-1)

        return cc_repre

    @overrides
    def forward(
            self,  # type: ignore
            diff_add_char: TextFieldTensors,
            diff_del_char: TextFieldTensors,
            diff_add_token: TextFieldTensors,
            diff_del_token: TextFieldTensors,
            diff_op: TextFieldTensors = None,
            msg: TextFieldTensors = None,
            additional_feature: Optional[torch.Tensor] = None,
            add_token_op_mask: Optional[torch.Tensor] = None,
            del_token_op_mask: Optional[torch.Tensor] = None,
            add_char_op_mask: Optional[torch.Tensor] = None,
            del_char_op_mask: Optional[torch.Tensor] = None,
            add_line_idx: Optional[torch.Tensor] = None,
            del_line_idx: Optional[torch.Tensor] = None,
            label: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        self.debug_forward_count += 1

        cc_repre_char = self.forward_features1(diff_add_char, diff_del_char, diff_op, msg, additional_feature,
                                               add_char_op_mask, del_char_op_mask,
                                               add_line_idx, del_line_idx)

        cc_repre_token = self.forward_features(diff_add_token, diff_del_token, diff_op, msg, additional_feature,
                                               add_token_op_mask, del_token_op_mask,
                                               add_line_idx, del_line_idx)

        cc_repre = cc_repre_char + cc_repre_token
        # cc_repre =  cc_repre_char
        # cc_repre = torch.cat([cc_repre_char, cc_repre_token], dim=-1)

        # print(cc_repre_char.shape)  #(8,768)
        # print(cc_repre_token.shape) #(8,768)
        # 20251022 wjk>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # cc_repre_char = cc_repre_char.unsqueeze(1)
        # cc_repre_token = cc_repre_token.unsqueeze(1)
        # output = self.model(cc_repre_token, cc_repre_char)
        # cc_repre = output.squeeze(1)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        probs, pred_idxes = self._classifier(cc_repre)
        result = {'probs': probs, 'pred': pred_idxes}

        if label is not None:
            loss = self._loss_func(probs, label)
            result['loss'] = loss
            update_metric(self._metric, pred_idxes, probs, label)

        return result

    def _ensure_tensor(self, token_ids):
        """
        确保 token_ids 是 tensor 并生成 mask
        token_ids: list 或 tensor，patch 长度可能不同
        返回 tensor [B,F,P,L], mask [B,F,P,L]
        """
        # 如果是 tensor 且维度是 4D
        if isinstance(token_ids, torch.Tensor) and token_ids.dim() == 4:
            mask = (token_ids != 0).float()
            return token_ids.long(), mask

        # token_ids 是 list 或 int
        B = len(token_ids)
        F_ = len(token_ids[0])
        P = len(token_ids[0][0])

        # 找到最大 patch 长度
        max_L = 0
        for b in range(B):
            for f in range(F_):
                for p in range(P):
                    patch = token_ids[b][f][p]
                    if isinstance(patch, int):
                        patch = [patch]
                        token_ids[b][f][p] = patch
                    max_L = max(max_L, len(patch))

        tensor_ids = torch.zeros(B, F_, P, max_L, dtype=torch.long)
        mask = torch.zeros(B, F_, P, max_L, dtype=torch.float)

        for b in range(B):
            for f in range(F_):
                for p in range(P):
                    patch = token_ids[b][f][p]
                    Lp = len(patch)
                    tensor_ids[b, f, p, :Lp] = torch.tensor(patch, dtype=torch.long)
                    mask[b, f, p, :Lp] = 1.0

        return tensor_ids, mask

    def embed_diff_add3(self, diff, patch_op_mask=None):
        """
        diff['code_tokens']['token_ids']: [B,F,P] 每个 patch 是 list[int]
        diff['code_tokens']['mask']: 可选
        """
        # ---------- Step 0: 转 tensor + mask ----------
        token_ids, token_mask = self._ensure_tensor(diff['code_tokens']['token_ids'])
        B, F_val, P, L = token_ids.shape

        # patch-level valid mask
        patch_valid_mask = (token_mask.sum(dim=-1) > 0).long()  # [B,F,P]

        # ---------- Step 1: reshape L -> H x W ----------
        patch_H = math.ceil(L / self.patch_W)
        pad_len = patch_H * self.patch_W - L

        # 对 token_ids 和 token_mask 进行右侧 padding
        x_padded = F.pad(token_ids, (0, pad_len))  # [B,F,P,H*W]
        m_padded = F.pad(token_mask, (0, pad_len))  # [B,F,P,H*W]

        # ---------- Step 1b: reshape成2D用于CNN ----------
        N = B * F_val * P
        x_2d = x_padded.view(N, 1, patch_H, self.patch_W).float()  # [N,1,H,W]
        m_2d = m_padded.view(N, 1, patch_H, self.patch_W).float()

        # ---------- Step 2: 2D CNN ----------
        feat = self.patch_conv2d(x_2d)  # [N,D,H,W]
        feat = feat * m_2d[:, :, :feat.size(2), :feat.size(3)]  # mask padding

        # ---------- Step 3: pooling ----------
        feat = self.patch_pool2d(feat).view(N, -1)  # [N,D]

        # 对无效 patch 清零
        valid_flat = patch_valid_mask.view(-1)
        feat[valid_flat == 0] = 0.0

        # ---------- Step 4: reshape回原维度 ----------
        patch_emb = feat.view(B, F_val, P, -1)  # [B,F,P,D]

        B, F_val, P_emb, D = patch_emb.shape  # [8, 4, 70, 768]
        if patch_op_mask is not None:
            if isinstance(patch_op_mask, list):
                patch_op_mask = torch.tensor(patch_op_mask, dtype=torch.float)

            # 裁剪或 pad 到 P_emb
            P_mask = patch_op_mask.shape[-1]
            if P_mask > P_emb:
                patch_op_mask = patch_op_mask[:, :, :P_emb]  # 裁剪多余的 patch
            elif P_mask < P_emb:
                pad_size = P_emb - P_mask
                patch_op_mask = F.pad(patch_op_mask, (0, pad_size), "constant", 0)

        result = {
            "source_mask": patch_valid_mask,  # [B,F,P]
            "encoder_outputs": patch_emb  # [B,F,P,D]
        }

        return result, patch_op_mask

    def embed_diff_add2(self, diff_add, patch_op_mask, no_grad: bool = True, max_batch_patches: int = 128):
        """
        高效批处理版：一次性批量嵌入所有 patch，避免循环调用嵌入模型。
        支持自动分块防止显存溢出（max_batch_patches）。

        输入:
            diff_add: dict, code_tokens -> {token_ids, type_ids, mask}
                      shapes: token_ids/mask/type_ids: [B, F, P, L]
            patch_op_mask: 原始 patch-level mask [B, F, P_old]
            no_grad: 是否使用 torch.no_grad() 节省显存
            max_batch_patches: 每次最多送入嵌入模型的 patch 数量（控制显存）

        输出:
            result: {
                "source_mask": patch_valid_mask,   # [B, F, P]
                "encoder_outputs": patch_emb       # [B, F, P, D]
            }
            patch_op_mask: reshape 后与 encoder_outputs 对齐 [B, F, P]
        """
        patch_ids = diff_add['code_tokens']['token_ids']  # [B, F, P, L]
        patch_mask = diff_add['code_tokens']['mask']  # [B, F, P, L]
        patch_types = diff_add['code_tokens'].get('type_ids', None)

        B, F, P, L = patch_ids.shape
        device = patch_ids.device
        D = self._code_embedder.get_output_dim() if hasattr(self._code_embedder, 'get_output_dim') else 768

        # 1️⃣ 展平为 [B*F*P, L]
        flat_token_ids = patch_ids.view(-1, L)
        flat_mask = patch_mask.view(-1, L)
        flat_type_ids = patch_types.view(-1, L) if patch_types is not None else None

        # 2️⃣ 过滤掉空 patch（mask 全 0 或 token 全 0）
        valid_mask = (flat_mask.sum(dim=1) > 0) & (flat_token_ids.max(dim=1).values > 0)
        valid_token_ids = flat_token_ids[valid_mask]
        valid_mask_ids = flat_mask[valid_mask]
        if flat_type_ids is not None:
            valid_type_ids = flat_type_ids[valid_mask]

        N_valid = valid_token_ids.size(0)
        if N_valid == 0:
            patch_emb = torch.zeros(B, F, P, D, device=device)
            patch_valid_mask = torch.zeros(B, F, P, device=device, dtype=torch.long)
            result = {
                "source_mask": patch_valid_mask,
                "encoder_outputs": patch_emb
            }
            if patch_op_mask is not None:
                patch_op_mask = patch_op_mask.view(-1)[:B * F * P].view(B, F, P)
            return result, patch_op_mask

        # 3️⃣ 分块处理所有 patch
        all_patch_vecs = []
        with torch.no_grad() if no_grad else torch.enable_grad():
            for start in range(0, N_valid, max_batch_patches):
                end = min(start + max_batch_patches, N_valid)
                batch_input = {
                    'code_tokens': {
                        'token_ids': valid_token_ids[start:end],
                        'mask': valid_mask_ids[start:end]
                    }
                }
                if flat_type_ids is not None:
                    batch_input['code_tokens']['type_ids'] = valid_type_ids[start:end]

                emb_all = self._code_embedder(batch_input)  # [n, L, D]

                # 1.取均值
                # valid_float = valid_mask_ids[start:end].float()
                # sum_emb = (emb_all * valid_float.unsqueeze(-1)).sum(dim=1)
                # len_valid = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
                # patch_vecs = sum_emb / len_valid  # [n, D]

                # 2.取最大值
                # 对 mask=0 的位置填充一个极小值，防止影响最大值
                masked_emb = emb_all.masked_fill(valid_mask_ids[start:end].unsqueeze(-1) == 0, float('-inf'))
                patch_vecs, _ = masked_emb.max(dim=1)  # [n, D]
                # 若整行全为 -inf（即mask全0），替换为0向量（防NaN）
                patch_vecs[patch_vecs.isnan()] = 0.0

                # 后续可以尝试把max和mean进行concat

                # 3.Attention Pooling方式
                # with torch.enable_grad():
                #     # Step 1: 线性 + tanh
                #     attn_hidden = torch.tanh(self.patch_attn_W(emb_all))  # [n, L, D]

                #     # Step 2: 投影成标量
                #     attn_scores = self.patch_attn_v(attn_hidden).squeeze(-1)  # [n, L]

                #     # Step 3: mask padding
                #     attn_scores = attn_scores.masked_fill(valid_mask_ids[start:end] == 0, float('-inf'))

                #     # Step 4: softmax
                #     attn_weights = torch.softmax(attn_scores, dim=1)  # [n, L]

                #     # Step 5: 加权求和
                #     patch_vecs = torch.sum(emb_all * attn_weights.unsqueeze(-1), dim=1)  # [n, D]

                # 4.BiLSTM方式聚合
                # with torch.enable_grad():
                #     lengths = valid_mask_ids[start:end].sum(dim=1).cpu()
                #     packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                #         emb_all, lengths=lengths, batch_first=True, enforce_sorted=False
                #     )
                #     # 建模 patch 内字符的顺序依赖，得到每个 patch 的序列表示。
                #     lstm_out, (h_n, c_n) = self.patch_bilstm(packed_input)

                #     # h_n: [2, n, hidden] -> 拼接正向和反向
                #     # h_n[0]: 正向最后一个时间步的 hidden → [n, hidden]
                #     # h_n[1]: 反向最后一个时间步的 hidden → [n, hidden]
                #     patch_vecs = torch.cat([h_n[0], h_n[1]], dim=-1)  # [n, D]

                #     # 可选：再映射回原始 D（如果 hidden_size 不等于 D//2）
                #     # patch_vecs = self.patch_bilstm_fc(patch_vecs)  # [n, D]

                # 5. gated pooling  这个方式还有些问题
                # # 计算 gate
                # gate = torch.sigmoid(self.patch_gate_W(emb_all))  # [n, L, D]
                # # mask padding
                # gate = gate * valid_mask_ids[start:end].unsqueeze(-1).float()  # mask=0的token权重为0
                # # 加权求和
                # patch_vecs = torch.sum(emb_all * gate, dim=1) / (gate.sum(dim=1, keepdim=True).clamp(min=1e-6))

                all_patch_vecs.append(patch_vecs)

        # 4️⃣ 合并结果 [N_valid, D]
        all_patch_vecs = torch.cat(all_patch_vecs, dim=0)

        # 5️⃣ 填回完整 [B*F*P, D]
        patch_emb_flat = torch.zeros(B * F * P, D, device=device)
        patch_emb_flat[valid_mask] = all_patch_vecs
        patch_emb_flat[~valid_mask] = 0.0

        # 6️⃣ reshape 到 [B, F, P, D]
        patch_emb = patch_emb_flat.view(B, F, P, D)

        # 7️⃣ patch_valid_mask
        patch_valid_mask = valid_mask.view(B, F, P).to(device, dtype=torch.long)

        # 8️⃣ 对齐 patch_op_mask
        if patch_op_mask is not None:
            patch_op_mask = patch_op_mask.view(-1)[:B * F * P].view(B, F, P)

        result = {
            "source_mask": patch_valid_mask,  # [B, F, P]
            "encoder_outputs": patch_emb  # [B, F, P, D]
        }

        return result, patch_op_mask

    # 完整可运行的字符级嵌入表示方法  速度比较慢
    # def embed_diff_add2(self, diff_add, patch_op_mask, no_grad: bool = True):
    #     """
    #     对每个 patch 单独进行嵌入表示，保证 patch 对齐，同时可选择 no_grad 节省显存。

    #     输入:
    #         diff_add: dict, code_tokens -> {token_ids, type_ids, mask}
    #                   shapes: token_ids/mask/type_ids: [B, F, P, L]
    #         patch_op_mask: 原始 patch-level mask，可能与 patch 数量不匹配 [B, F, P_old]
    #         no_grad: 是否使用 torch.no_grad() 节省显存

    #     输出:
    #         result: {
    #             "source_mask": patch_valid_mask,   # [B, F, P] 记录 patch 是否有效
    #             "encoder_outputs": patch_emb       # [B, F, P, D]
    #         }
    #         patch_op_mask: reshape 后与 encoder_outputs 对齐 [B, F, P]
    #     """
    #     patch_ids = diff_add['code_tokens']['token_ids']  # [B, F, P, L]
    #     patch_types = diff_add['code_tokens'].get('type_ids', None)
    #     patch_mask = diff_add['code_tokens']['mask']      # [B, F, P, L]

    #     B, F, P, L = patch_ids.shape
    #     device = patch_ids.device
    #     D = self._code_embedder.get_output_dim() if hasattr(self._code_embedder, 'get_output_dim') else 768

    #     patch_emb_list = []
    #     patch_valid_list = []

    #     for b in range(B):
    #         file_embs = []
    #         file_valids = []
    #         for f in range(F):
    #             patch_embs = []
    #             patch_valids = []
    #             for p in range(P):
    #                 token_ids = patch_ids[b, f, p].unsqueeze(0)   # [1, L]
    #                 mask = patch_mask[b, f, p].unsqueeze(0)       # [1, L]
    #                 type_ids = patch_types[b, f, p].unsqueeze(0) if patch_types is not None else None
    #                 # 🔍 检查是否为空 patch 或 mask 全 0
    #                 if mask.sum() == 0 or (token_ids == 0).all() or token_ids.numel() == 0 :   # 说明没有有效字符
    #                     # 创建一个全零向量代替
    #                     patch_vec = torch.zeros(1, D, device=device)
    #                     patch_embs.append(patch_vec)
    #                     patch_valids.append(0)
    #                     continue
    #                 # 🚨 再增加一层防御：token_ids 中出现非法字符(空字符串映射0)
    #                 if token_ids.max() == 0:  # 表示 tokenizer 全部映射为空token
    #                     patch_vec = torch.zeros(1, D, device=device)
    #                     patch_embs.append(patch_vec)
    #                     patch_valids.append(0)
    #                     continue
    #                 if (mask.sum() == 0 or (token_ids == 0).all() or token_ids.numel() == 0):
    #                     print(f"[Warning] Empty patch at batch={b}, file={f}, patch={p}")

    #                 text_field_input = {'code_tokens': {'token_ids': token_ids, 'mask': mask}}
    #                 if type_ids is not None:
    #                     text_field_input['code_tokens']['type_ids'] = type_ids

    #                 # 单 patch forward
    #                 if no_grad:
    #                     with torch.no_grad():
    #                         emb = self._code_embedder(text_field_input)  # [1, L, D]
    #                 else:
    #                     emb = self._code_embedder(text_field_input)

    #                 # patch-level 平均池化
    #                 valid_float = mask.float()                     # [1, L]
    #                 sum_emb = (emb * valid_float.unsqueeze(-1)).sum(dim=1)  # [1, D]
    #                 len_valid = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
    #                 patch_vec = sum_emb / len_valid
    #                 if len_valid.item() == 0:
    #                     patch_vec[0] = 0.0

    #                 patch_embs.append(patch_vec)                  # [1, D]
    #                 patch_valids.append(int(mask.any().item()))   # 1/0

    #             # 合并该文件所有 patch
    #             file_embs.append(torch.cat(patch_embs, dim=0))      # [P, D]
    #             file_valids.append(torch.tensor(patch_valids, device=device))  # [P]

    #         # 合并 batch
    #         patch_emb_list.append(torch.stack(file_embs, dim=0))     # [F, P, D]
    #         patch_valid_list.append(torch.stack(file_valids, dim=0))  # [F, P]

    #     # 合并所有 batch
    #     patch_emb = torch.stack(patch_emb_list, dim=0)       # [B, F, P, D]
    #     patch_valid_mask = torch.stack(patch_valid_list, dim=0)  # [B, F, P]

    #     # reshape patch_op_mask 与 patch_emb对齐，保留前面的有效 patch
    #     if patch_op_mask is not None:
    #         patch_op_mask = patch_op_mask.view(-1)[:B*F*P].view(B, F, P)

    #     result = {
    #         "source_mask": patch_valid_mask,
    #         "encoder_outputs": patch_emb
    #     }

    #     return result, patch_op_mask

    def filter_and_embed(self, token_ids, type_ids, mask, embedder: TextFieldEmbedder, ):
        B, S, L = token_ids.shape  # batch size, num_subseq, seq_len

        # 展平为 [B*S, L]
        flat_token_ids = token_ids.view(B * S, L)
        flat_type_ids = type_ids.view(B * S, L)
        flat_mask = mask.view(B * S, L)

        # 过滤出有效子序列（至少有一个非 padding）
        valid_mask = flat_mask.sum(dim=1) > 0  # [B*S]
        valid_token_ids = flat_token_ids[valid_mask]
        valid_type_ids = flat_type_ids[valid_mask]
        valid_mask_ids = flat_mask[valid_mask]
        result = {
            'code_tokens': {'token_ids': valid_token_ids, 'mask': valid_mask_ids, 'type_ids': valid_type_ids}
        }

        # 仅对有效子序列进行嵌入
        embeddings = embedder(
            result
        )  # [N_valid, L, D]

        # 初始化输出张量为零（防止 nan），再将有效结果放进去
        D = embeddings.size(-1)
        all_embeddings = torch.zeros(B * S, L, D, device=token_ids.device)
        all_embeddings[valid_mask] = embeddings
        all_embeddings[~valid_mask] = 1e-5  # 替代全0，避免 LayerNorm 等数值出错
        # reshape 回 [B, S, L, D]
        return all_embeddings.view(B, S, L, D)

    def _encode_text_field(self, tokens: Dict[str, Dict[str, torch.Tensor]],
                           embedder: TextFieldEmbedder,
                           encoder: Seq2SeqEncoder) -> Dict[str, torch.Tensor]:
        '''
        Same as _encode() method of ComposedSeq2Seq.
        '''
        # Note: It assumes there is only one tensor in the dict.
        dim_num = -1
        for k1, d in tokens.items():
            for k2, t in d.items():
                dim_num = len(t.size())
                break
        # adapt multi-dimensional input
        num_wrapping_dim = dim_num - 2

        # shape: (batch_size, max_input_sequence_length)
        seq_mask = util.get_text_field_mask(tokens, num_wrapping_dim)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)

        # embedded_features = embedder(tokens, num_wrapping_dims=num_wrapping_dim)
        embedded_features = self.filter_and_embed(tokens['code_tokens']['token_ids'], tokens['code_tokens']['type_ids'],
                                                  tokens['code_tokens']['mask'], embedder)

        # vocab_size = embedder.get_input_embeddings().num_embeddings
        # token_ids = tokens['token_ids'].clamp(0, vocab_size - 1)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = encoder(embedded_features, seq_mask)
        return {
            "source_mask": seq_mask,
            "encoder_outputs": encoder_outputs
        }

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self._metric is not None:
            metric = self._metric.get_metric(reset)
            # no metric name returned, use its class name instead
            if type(metric) != dict:
                metric_name = self._metric.__class__.__name__
                metric = {metric_name: metric}
            metrics.update(metric)
        return metrics
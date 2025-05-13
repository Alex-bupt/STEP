"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from Qformer.models.base_model import all_gather_with_grad, concat_all_gather
from Qformer.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
)

from torch.nn import functional as F


class EntityQformer(Blip2Base):

    def __init__(
            self,
            entity_size=768,
            num_query_token=48,
            cross_attention_freq=2,
            embed_dim=768,
            max_txt_len=256,
            use_margin_loss=True,  # 是否启用 margin loss 模块
            margin_loss_weight=0.1,  # margin loss 的权重
            margin=0.1,  # margin 超参数
            use_aux_etm=True,  # 是否启用辅助 ITM 任务
            aux_etm_weight=0.2,  # 辅助 ITM loss 的权重
            triplet_weight=0.2,
            use_mem=True,
            mem_ratio=0.15,
            mem_weight=0.2,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, entity_size, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.entity_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        # Margin loss 相关参数
        self.use_margin_loss = use_margin_loss
        self.margin_loss_weight = margin_loss_weight
        self.margin = margin

        # 辅助 ITM 任务相关参数
        self.use_aux_etm = use_aux_etm
        self.aux_etm_weight = aux_etm_weight

        self.triplet_weight = triplet_weight  # Triplet loss 权重

        self.local_text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.use_mem = use_mem
        self.mem_ratio = mem_ratio
        self.mem_weight = mem_weight
        # 将重建向量映射回原空间
        self.mem_pred = nn.Linear(embed_dim, embed_dim)

    def forward(self, entity, context, rec_feats):

        bs = entity.size(0)  # batch size
        entity_embeds = entity  # [B, L, D]

        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(entity.device)
        # 将 query_tokens 扩展到 batch 维度，形状 [B, Q, D]
        query_tokens = self.query_tokens.expand(bs, -1, -1)

        # === 分支1：Entity attends KG (原始分支) ===
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            output_hidden_states=True,
            return_dict=True,
            is_decoder=False
        )

        last_hidden = query_output.hidden_states[-1]

        # 投影和归一化后，得到 [B, Q, D] 的实体特征
        entity_feats_1 = F.normalize(self.entity_proj(last_hidden), dim=-1)

        # === 文本编码 ===
        text_tokens = self.tokenizer(
            context,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        text_tokens = {k: v.to(entity.device) for k, v in text_tokens.items()}
        text_output = self.Qformer.bert(
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
            is_decoder=False
        )

        text_last_hidden = text_output.hidden_states[-1]

        # 获取[CLS] token代表的全局文本信息， shape [B, D]
        text_feat = F.normalize(
            self.text_proj(text_last_hidden[:, 0, :]), dim=-1
        )

        # === 分支2：Entity attends Text (反向分支) ===
        entity_as_query = entity_feats_1  # [B, Q, D]
        reverse_output = self.Qformer.bert(
            query_embeds=entity_as_query,
            encoder_hidden_states=text_last_hidden,
            encoder_attention_mask=text_tokens["attention_mask"],
            return_dict=True,
            output_hidden_states=True,
            is_decoder=False
        )

        reverse_last_hidden = reverse_output.hidden_states[-1]

        # 得到反向注意力输出，同样经过投影和归一化，得到 [B, Q, D]
        entity_feats_2 = F.normalize(self.entity_proj(reverse_last_hidden), dim=-1)

        # === 融合两路输出 ===
        entity_feats = (entity_feats_1 + entity_feats_2) / 2

        loss_total = 0.0  # 原有主、aux 任务之和

        if self.use_aux_etm and rec_feats is not None:
            # 全局融合表示 entity_repr: [B, D]
            entity_repr = entity_feats.mean(dim=1)
            # 使用 cosine embedding loss，目标为 1（即希望余弦相似度越接近 1 越好）
            loss_aux = F.cosine_embedding_loss(entity_repr, rec_feats, torch.ones(bs, device=entity.device))
            loss_total = loss_total + self.aux_etm_weight * loss_aux

        # === Contrastive Loss with Batch-Hard Negative Mining ===
        sim_q2t = torch.matmul(entity_feats, text_feat.t().unsqueeze(0))  # [B, Q, B]
        sim_q2t = sim_q2t.permute(0, 2, 1)  # [B, B, Q]
        sim_i2t, _ = sim_q2t.max(dim=-1)  # [B, B]
        sim_i2t = sim_i2t / self.temp

        sim_matrix = torch.einsum('id,jqd->ijq', text_feat, entity_feats)  # [B, B, Q]
        sim_t2i, _ = sim_matrix.max(dim=-1)  # [B, B]
        sim_t2i = sim_t2i / self.temp

        # --- Batch-Hard Negative Mining ---
        mask = torch.eye(bs, dtype=torch.bool, device=entity.device)
        sim_i2t_masked = sim_i2t.masked_fill(mask, -float('inf'))
        hardest_neg_i2t, _ = sim_i2t_masked.max(dim=1)  # [B]
        positive_i2t = sim_i2t.diag()  # [B]

        sim_t2i_masked = sim_t2i.masked_fill(mask, -float('inf'))
        hardest_neg_t2i, _ = sim_t2i_masked.max(dim=1)  # [B]
        positive_t2i = sim_t2i.diag()  # [B]

        targets = torch.arange(bs, device=entity.device)
        loss_ce = (
                          F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                          F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                  ) / 2

        loss_margin = 0.0
        if self.use_margin_loss:
            loss_margin_i2t = F.relu(self.margin + hardest_neg_i2t - positive_i2t)
            loss_margin_t2i = F.relu(self.margin + hardest_neg_t2i - positive_t2i)
            loss_margin = (loss_margin_i2t.mean() + loss_margin_t2i.mean()) / 2

        loss_total = loss_total + loss_ce + self.margin_loss_weight * loss_margin

        if rec_feats is not None:
            # 1) 对每个样本把 Q 个 slot 平均，得到 [B, D] 的实体融合向量
            entity_repr = entity_feats.mean(dim=1)  # [B, D]
            # 2) 归一化
            entity_norm = F.normalize(entity_repr, dim=-1)  # [B, D]
            rec_norm = F.normalize(rec_feats, dim=-1)  # [B, D]

            # 3) 计算两两相似度矩阵 [B, B]
            sim = torch.matmul(entity_norm, rec_norm.t())  # [B, B]

            # 4) 正样本在对角线
            pos_sim = sim.diag()  # [B]

            # 5) 屏蔽对角线，得到最难负样本
            mask = torch.eye(bs, dtype=torch.bool, device=entity.device)
            sim_neg = sim.masked_fill(mask, -float('inf'))
            hardest_neg_sim, _ = sim_neg.max(dim=1)  # [B]

            # 6) Triplet Loss：max(0, neg − pos + margin)
            loss_triplet = F.relu(hardest_neg_sim - pos_sim + self.margin).mean()

            # 7) 加权累加
            loss_total = loss_total + self.triplet_weight * loss_triplet

        return entity_feats, text_feat, loss_total

        # B, Q, D = entity_feats.shape

        # ——— MEM：Masked Entity Modeling ———
        # loss_mem = torch.tensor(0.0, device=entity.device)
        # if self.use_mem:
        #     # 随机掩码部分 slot
        #     mask = (torch.rand(B, Q, device=entity.device) < self.mem_ratio)
        #     if mask.any():
        #         # 1) 构造 mask 后的 entity_feats_1
        #         masked_feats = entity_feats_1.clone()
        #         masked_feats[mask] = 0.0
        #
        #         # 2) 通过同样的反向分支（Text→Entity）尝试重建
        #         rec_output = self.Qformer.bert(
        #             query_embeds=masked_feats,  # [B, Q, H]
        #             encoder_hidden_states=text_last_hidden,
        #             encoder_attention_mask=text_tokens["attention_mask"],
        #             return_dict=True,
        #             output_hidden_states=True,
        #             is_decoder=False
        #         )
        #         rec_hidden = rec_output.hidden_states[-1]  # [B, Q, H]
        #         rec_proj = F.normalize(self.entity_proj(rec_hidden), -1)  # [B, Q, D]
        #         # 3) 预测头产出重建向量
        #         rec_recon = self.mem_pred(rec_proj)  # [B, Q, D]
        #         # 4) 对掩码位置做 MSE 重建损失
        #         loss_mem = F.mse_loss(rec_recon[mask], entity_feats_1[mask])
        #
        #     loss_total = loss_total + self.aux_etm_weight * loss_mem

        # # === 辅助任务：Entity-Text Matching (ETM) 分类任务 ===
        #
        # if rec_feats is not None:
        #     # 确保 rec_feats 归一化
        #     rec_feats_norm = F.normalize(rec_feats, dim=-1)
        #     # 计算每个文本表示与所有正样本的余弦相似度矩阵，形状 [B, B]
        #     cosine_sim = torch.matmul(text_feat, rec_feats_norm.transpose(0, 1))  # [B, B]
        #     # 正样本相似度：取对角线
        #     pos_sim = cosine_sim.diag()  # [B]
        #     # 将对角线屏蔽后，得到每个样本最难的负样本相似度
        #     mask = torch.eye(bs, dtype=torch.bool, device=entity.device)
        #     cosine_sim_neg = cosine_sim.masked_fill(mask, -1e9)
        #     hardest_neg_sim, _ = cosine_sim_neg.max(dim=1)
        #     # 计算 Triplet Loss：要求负样本相似度低于正样本相似度 margin 个单位
        #     loss_triplet = F.relu(hardest_neg_sim - pos_sim + self.aux_etm_weight * self.margin).mean()
        #     loss_total = loss_total + self.triplet_weight * loss_triplet


        # ##########################
        # # —— ETC 细粒度对齐 —— #
        # ##########################
        # # 1) 将 reverse_hidden 投影并归一化，得到每个 slot 的“局部文本表示”
        # # reverse_last_hidden: [B, Q, H]; H=hidden_size
        # local_text_feats = F.normalize(
        #     self.local_text_proj(reverse_last_hidden), dim=-1
        # )  # [B, Q, D]
        #
        # # 2) Flatten batch & slot 维度，形成 N = B*Q 个样本对
        # B, Q, D = entity_feats.size()
        # N = B * Q
        # ent_flat = entity_feats.view(N, D)  # [N, D]
        # txt_flat = local_text_feats.view(N, D)  # [N, D]
        #
        # # 3) 计算对比相似度矩阵
        # sim_mat = torch.matmul(ent_flat, txt_flat.t()) / self.temp  # [N, N]
        #
        # # 4) 正样本在对角线
        # targets = torch.arange(N, device=entity.device)
        #
        # # 5) InfoNCE 损失
        # loss_etc = F.cross_entropy(sim_mat, targets)
        #
        # # 6) 加权合并 ETC
        # alpha_etc = 0.2  # 可调超参
        # loss_total = loss_total + self.aux_etm_weight * loss_etc




        # text = context
        # entity_embeds = entity
        #
        # entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
        #     entity.device
        # )
        #
        # query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)
        #
        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=entity_embeds,
        #     encoder_attention_mask=entity_atts,
        #     use_cache=True,
        #     return_dict=True,
        # )
        #
        # entity_feats = F.normalize(
        #     self.entity_proj(query_output.last_hidden_state), dim=-1
        # )
        #
        # text_tokens = self.tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(entity.device)
        #
        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        #
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        #
        #
        # ##============== entity-text Contrastive ===================###
        # entity_feats_all = concat_all_gather(
        #     entity_feats
        # )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        # text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]
        #
        # sim_q2t = torch.matmul(
        #     entity_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        # ).squeeze()
        # # [batch_size, batch_size*num_gpu, num_query_tokens]
        #
        # # entity-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp
        #
        # # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), entity_feats_all.permute(0, 2, 1)
        # ).squeeze()
        #
        # # text-entity similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        #
        # rank = dist.get_rank()
        # bs = entity.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        #     entity.device
        # )
        #
        # loss_etc = (
        #                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        #            ) / 2

        ###============== Entity-text Matching ===================###
        # text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        # text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        # entity_embeds_world = all_gather_with_grad(entity_embeds)
        #
        # with torch.no_grad():
        #     sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
        #     sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
        #
        #     weights_t2i = F.softmax(sim_t2i, dim=1)
        #     weights_i2t = F.softmax(sim_i2t, dim=1)
        #
        #     # 修正 weights_t2i 和 weights_i2t 的 inf 和 nan
        #     weights_t2i = torch.where(torch.isnan(weights_t2i) | torch.isinf(weights_t2i),
        #                               torch.tensor(0.0, device=weights_t2i.device), weights_t2i)
        #     weights_i2t = torch.where(torch.isnan(weights_i2t) | torch.isinf(weights_i2t),
        #                               torch.tensor(0.0, device=weights_i2t.device), weights_i2t)
        #
        #     # 将负值裁剪为 0
        #     weights_t2i = torch.clamp(weights_t2i, min=0.0)
        #     weights_i2t = torch.clamp(weights_i2t, min=0.0)
        #
        #     # 确保概率分布的和为正值
        #     if (weights_t2i.sum(dim=1) == 0).any():
        #         weights_t2i += 1e-10
        #     if (weights_i2t.sum(dim=1) == 0).any():
        #         weights_i2t += 1e-10
        #
        # # select a negative entity for each text
        # entity_embeds_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        #     entity_embeds_neg.append(entity_embeds_world[neg_idx])
        # entity_embeds_neg = torch.stack(entity_embeds_neg, dim=0)
        #
        # # select a negative text for each entity
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_input_ids_world[neg_idx])
        #     text_atts_neg.append(text_attention_mask_world[neg_idx])
        #
        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)
        #
        # text_ids_all = torch.cat(
        #     [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        # )  # pos, pos, neg
        # text_atts_all = torch.cat(
        #     [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        #     dim=0,
        # )
        #
        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     entity_embeds.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)
        #
        # entity_embeds_all = torch.cat(
        #     [entity_embeds, entity_embeds_neg, entity_embeds], dim=0
        # )  # pos, neg, pos
        # entity_atts_all = torch.ones(entity_embeds_all.size()[:-1], dtype=torch.long).to(
        #     entity_embeds.device
        # )
        #
        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=entity_embeds_all,
        #     encoder_attention_mask=entity_atts_all,
        #     return_dict=True,
        # )
        #
        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)
        #
        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(entity_embeds.device)
        # loss_etm = F.cross_entropy(logits, itm_labels)
        #
        # loss = loss_etm + loss_etc

        # return entity_feats, text_feat, loss_etc

    def forward_entity(self, entity):
        entity_embeds = entity.unsqueeze(1)
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            return_dict=True,
        )

        entity_feats = self.entity_proj(query_output.last_hidden_state)

        cosine_sim = torch.matmul(entity_feats, entity.unsqueeze(-1)).squeeze(-1)  # [bs, 32]

        # 找到相似度最高的索引
        _, max_indices = torch.max(cosine_sim, dim=1)  # [bs]

        # 选择相似度最高的向量
        final_entity_feats = torch.stack([entity_feats[i, idx] for i, idx in enumerate(max_indices)],
                                         dim=0)  # [bs, 768]

        return final_entity_feats

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def user_embed_generate(self, entity_embeds, context):
        entity_embeds = entity_embeds
        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity_embeds.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            return_dict=True,
        )

        entity_feats = self.entity_proj(query_output.last_hidden_state)  # [bs, 32, 768]

        text_tokens = self.tokenizer(
            context,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity_embeds.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])  # [bs, 768]

        return entity_feats, text_feat

    def fuse_embeds(self, entity, text):
        entity_embeds = entity
        bs = entity_embeds.size(0)

        entity_atts = torch.ones(entity_embeds.size()[:-1], dtype=torch.long).to(
            entity.device
        )

        query_tokens = self.query_tokens.expand(entity_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=entity_embeds,
            encoder_attention_mask=entity_atts,
            use_cache=True,
            return_dict=True,
        )

        entity_feats = self.entity_proj(query_output.last_hidden_state)

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(entity.device)

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )

        text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])

        text_feat = text_feat.repeat(bs, 1)

        cosine_sim = torch.matmul(entity_feats, text_feat.unsqueeze(-1)).squeeze(-1)  # [bs, 32]

        # 找到相似度最高的索引
        _, max_indices = torch.max(cosine_sim, dim=1)  # [bs]

        # 选择相似度最高的向量
        final_entity_feats = torch.stack([entity_feats[i, idx] for i, idx in enumerate(max_indices)],
                                         dim=0)  # [bs, 768]

        return final_entity_feats

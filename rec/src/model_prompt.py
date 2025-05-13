import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
from Qformer.models.blip2_models.fusion_qformer import EntityQformer


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations + 1,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        # stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        # self.node_embeds.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.node_embeds)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

        self.Qformer = EntityQformer()

        self.Qformer.margin_loss_weight = 0.0
        self.Qformer.aux_etm_weight = 0.0
        self.Qformer.triplet_weight = 0.0

        self.max_epochs = 5

    def update_curriculum(self, epoch: int):
        """
        根据当前 epoch 动态更新 Qformer 中各任务的权重。
        5 个 epoch 划分为：
          - margin 从 0 线性升到 0.1
          - aux_etm 从 epoch=3 开始线性升到 0.3
          - triplet 从 epoch=4 开始线性升到 0.2
        """
        E = self.max_epochs
        # margin: 0 → 0.1
        w_margin = 0.1
        # aux_etm: 从 epoch 3 (index 2) 开始 → 0.2
        w_aux = 0.3 * max(0, min(1, epoch / E))
        # triplet: 从 epoch 4 (index 3) 开始 → 0.2
        w_triplet = 0.4 * max(0, min(1, epoch / E))

        self.Qformer.margin_loss_weight = w_margin
        self.Qformer.aux_etm_weight = w_aux
        self.Qformer.triplet_weight = w_triplet

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, context_str=None, rec_labels=None):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            rec_feats = entity_embeds[rec_labels]
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        # if entity_embeds is not None and token_embeds is not None:
        #     attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2,
        #                                                                          1)  # (batch_size, token_len, entity_len)
        #     attn_weights /= self.hidden_size

            entity_embeds, token_embeds, loss = self.Qformer(entity_embeds, context_str, rec_feats)
            if output_entity:
                prompt_embeds = entity_embeds
                prompt_len = 48
            else:
                prompt_embeds = entity_embeds
                prompt_len = 48

            # if output_entity:
            #     token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
            #     prompt_embeds = token_weights @ token_embeds + entity_embeds
            #     prompt_len = entity_len
            # else:
            #     entity_weights = F.softmax(attn_weights, dim=2)
            #     prompt_embeds = entity_weights @ entity_embeds + token_embeds
            #     prompt_len = token_len
        elif entity_embeds is not None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds, loss

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)

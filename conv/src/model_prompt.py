import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None, prompt_max_length = 50, n_examples = 3, entity_hidden_size =  128
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        self.pad_entity_id = 31161
        self.prompt_max_length = prompt_max_length
        self.n_examples = n_examples

        # entity_hidden_size = hidden_size // 2
        entity_hidden_size = entity_hidden_size
        # entity_hidden_size = hidden_size
        self.kg_encoder = RGCNConv(n_entity, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
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
        self.prompt_proj2 = nn.Linear(hidden_size, hidden_size)

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

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        # entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.kg_encoder(None, self.edge_index, self.edge_type)
        # entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        # entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, token_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False, retrieved_entity_ids = None, word_embeddings = None, mapping = True, context_input_embeddings = None, attention_mask = None):

        batch_size, entity_embeds, entity_len, retrieved_vector, token_len = None, None, None, None, None
        retrieved_entity_embeds = None
        
        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds_all = self.get_entity_embeds()
            entity_embeds = entity_embeds_all[entity_ids]  # (batch_size, entity_len, hidden_size)
            ### entity selection step
            #########################

        if retrieved_entity_ids is not None:
            # entity_embeds = self.get_entity_embeds()
            retrieved_entity_embeds = entity_embeds_all[retrieved_entity_ids]

        # # return entity_embeds
        # if token_embeds is not None:
        #     batch_size, token_len = token_embeds.shape[:2]
        #     token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
        #     token_embeds = self.token_proj2(token_embeds)

        # if entity_embeds is not None and token_embeds is not None:
        #     prompt_embeds = entity_embeds
        #     prompt_len = entity_len

        # elif entity_embeds is not None:
        #     prompt_embeds = entity_embeds
        #     prompt_len = entity_len
        # else:
        #     prompt_embeds = token_embeds
        #     prompt_len = token_len

        # if self.n_prefix_rec is not None and use_rec_prefix:
        #     prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
        #     prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)

        #     ### prefix and demonstrations.
        #     ### for item recommendation task
        #     prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        #     prompt_len += self.n_prefix_rec

        # if self.n_prefix_conv is not None and use_conv_prefix:
        #     prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
        #     prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)

        #     #### prefix and the neural demonstrations
        #     # prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        #     prompt_embeds = prefix_embeds
        #     prompt_len += self.n_prefix_conv

        ##### semantic space mapping
        ##### get the prompt tokens
        if not output_entity:
            assert token_embeds.shape[0] // self.n_examples
            #### get the prompt embeddings
            prompt_embeds = token_embeds[:, :self.prompt_max_length, :]
            # prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
            ##### shape = [bs, prompt_length * n_examples, hidden_size]
            try:
                prompt_embeds = prompt_embeds.contiguous().view(batch_size, self.n_examples, self.prompt_max_length, self.hidden_size)
                prompt_embeds = prompt_embeds.view(batch_size, self.n_examples * self.prompt_max_length, self.hidden_size)
                # prompt_embeds = prompt_embeds.contiguous().view(batch_size, self.n_examples, self.prompt_max_length, word_embeddings.shape[0])
                # prompt_embeds = prompt_embeds.view(batch_size, self.n_examples * self.prompt_max_length, word_embeddings.shape[0])
        
                # word_embeddings = word_embeddings.contiguous().view(batch_size, self.n_examples, -1, self.hidden_size)
                # word_embeddings = word_embeddings.view(batch_size, -1, self.hidden_size)
                pass
            except:
                print(token_embeds.shape)
                print(batch_size, self.n_examples, self.prompt_max_length)
                print(prompt_embeds.shape)
                assert 1==0
            if mapping:
                ### [bs, prompt_length, hidden_size]
                ### [bs, retrived_gen_length, hidden_size] x [bs, retrieved_gen_length, hidden_size]
                # affinity_scores = self.cross_attn(prompt_embeds) @ word_embeddings.permute(0,2,1)
                affinity_scores = self.cross_attn(prompt_embeds) @ word_embeddings.T
                affinity_scores = affinity_scores / self.hidden_size
                prompt_embeds = torch.softmax(affinity_scores, dim =-1) @ word_embeddings
                # prompt_embeds = torch.softmax(prompt_embeds, dim =-1) @ word_embeddings

            ### create attention_mask
            prompt_attention_mask = torch.ones((prompt_embeds.shape[0], self.n_examples * self.prompt_max_length)).to(prompt_embeds.device)
            # prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
            # prompt_embeds = self.prompt_proj2(prompt_embeds)
            # # prompt_embeds = prompt_embeds.reshape(
            #     batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
            # ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

            ##### get the input embeddings for the generation model
            context_input_embeddings = torch.cat([prompt_embeds, context_input_embeddings], dim = 1)
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim =1)
            assert context_input_embeddings.shape[1] == attention_mask.shape[1]

            # return entity_embeds
            return context_input_embeddings, attention_mask, retrieved_vector, retrieved_entity_embeds
        else:
            return entity_embeds, entity_embeds_all, retrieved_entity_embeds

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

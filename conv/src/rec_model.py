import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionSeq(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)  # (batch, seq_len, 1)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            batch_mask = torch.sum((mask == False), -1).bool().float().unsqueeze(-1)  # for all padding one, the mask=0
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1)
        # (batch, dim)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)


class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = 0.8 * input1 + (1 - 0.8) * input2  # (bs, dim)
        return gated_emb

class DemonstrationRecModel(nn.Module):

    def __init__(self, entity_dim, word_dim = 768, n_entities = 31162 ):
        super(DemonstrationRecModel,self ).__init__()

        self.entity_dim = entity_dim
        self.word_dim = word_dim
        self.hidden_size = entity_dim
        self.pad_entity_id = 31161
        self.attn_layer= SelfAttentionSeq(self.hidden_size, self.hidden_size)
        self.attn_layer_2 = SelfAttentionSeq(self.hidden_size, self.hidden_size)

        self.rec_bias = nn.Linear( self.hidden_size, n_entities)
        self.rec_criterion = nn.CrossEntropyLoss()

        self.gate = GateLayer(entity_dim)
        self.entity_projection = nn.Linear(word_dim, entity_dim)

    def forward(self, token_embeds, entity_embeds, all_entity_embeds, retrieved_entity_ids = None, retrieved_entity_embeds = None, entity_ids = None, rec_labels = None):
        entity_mask = (entity_ids == self.pad_entity_id)
        user_vector = self.attn_layer(entity_embeds, mask = entity_mask.int())

        retrieved_mask = (retrieved_entity_ids == self.pad_entity_id)
        retrieved_vector = self.attn_layer_2(retrieved_entity_embeds, retrieved_mask )
        # gated_embed = self.gate(user_vector, retrieved_vector)
        word_vector = self.entity_projection(token_embeds[:, 0, :])

        user_vector = 0.8 * user_vector + 0.2 * word_vector
        rec_scores = F.linear(user_vector, all_entity_embeds, self.rec_bias.bias)
        if rec_labels is not None:
            rec_loss = self.rec_criterion(rec_scores, rec_labels)
            return rec_scores, rec_loss
        return rec_scores
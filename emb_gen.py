import torch
import torch.nn as nn
from collections import namedtuple


class EmbeddingGenerator(nn.Module):

    def __init__(
            self,
            input_hidden_size,
            num_attention_heads,
            output_hidden_size,
            num_layers=1,
    ):
        super().__init__()
        self.input_hidden_size = input_hidden_size
        self.num_attention_heads = num_attention_heads
        self.output_hidden_size = output_hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_hidden_size,
            nhead=self.num_attention_heads,
            activation='relu',
            batch_first=True,
        )
        self.num_layers = num_layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.input_hidden_size)

        self.input_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)
        self.output_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)

    def forward(self, inputs, attn_mask):
        out = self.encoder(inputs, src_key_padding_mask=~attn_mask.bool())
        out = self.norm(out)

        out = torch.sum(out * attn_mask.unsqueeze(-1), dim=1) / torch.sum(attn_mask, dim=-1, keepdim=True)

        out = torch.mean(out, dim=0, keepdim=True)

        inp_embeds = self.input_emb_head(out)
        out_embeds = self.output_emb_head(out)

        return inp_embeds, out_embeds


EmbGener = namedtuple('EmbGener', ['mlm_tokenizer', 'mlm', 'model', 'token_id'])
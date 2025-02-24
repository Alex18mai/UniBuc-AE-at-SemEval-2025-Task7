import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch import Tensor


class Embedder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)


    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    

    def forward(self, **kwargs):
        outputs = self.model(**kwargs).last_hidden_state
        embeddings = self.average_pool(outputs, kwargs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
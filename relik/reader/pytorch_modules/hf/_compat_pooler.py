# PoolerEndLogits copied from transformers 4.41.2 modeling_utils.py (removed in transformers 5). Apache-2.0.
from typing import Optional
import torch
from torch import nn
from transformers import PretrainedConfig

class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x



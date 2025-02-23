from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .wrapper import moba_layer
from .moba_naive import moba_attn_varlen_naive
from .config import MoBAConfig
import os

# Force MoBA to use the CPU-compatible `moba_naive` implementation
os.environ["MOBA_ATTENTION_TYPE"] = "moba_naive"

def register_moba(cfg: MoBAConfig):
    """
    Register MoBA's naive attention implementation (CPU-compatible).
    """
    ALL_ATTENTION_FUNCTIONS["moba_naive"] = partial(moba_layer, moba_attn_varlen_naive, cfg)

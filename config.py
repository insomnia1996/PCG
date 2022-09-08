'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import torch
class GPT2Config(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50257,
            eosid=50256,
            padid=50256,
            bosid=50256,
            field_vocab=2756,
            n_positions=1024,
            n_ctx=1024,
            hidden_size=768,
            position_vocab=31,
            n_layer=12,
            n_head=12,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            max_len=85,
            top_k=4,
            top_p=0.9,
            use_coverage=True,
            use_copy_gate=True,
            copy_gate_penalty=0.7,
            coverage_penalty=0.02,
            dropout=0.2,
            deviceid=0

    ):

        self.vocab_size = vocab_size_or_config_json_file
        self.field_vocab = field_vocab
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.hidden_size = hidden_size
        self.position_vocab = position_vocab
        self.pos_size = hidden_size
        self.field_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.max_length=max_len
        self.attention=True
        self.top_k = top_k
        self.top_p = top_p
        self.device =torch.device(("cuda:%d"% deviceid) if torch.cuda.is_available() else "cpu")
        self.eos=eosid
        self.pad=padid
        self.bos=bosid
        self.use_coverage = use_coverage
        self.use_copy_gate = use_copy_gate
        self.copy_gate_penalty = copy_gate_penalty
        self.coverage_penalty = coverage_penalty
        self.rand_unif_init_mag = 0.02
        self.trunc_norm_init_std = 1e-4


class BartConfig(object):
    def __init__(
            self,
            vocab_size_or_config_json_file=50265,
            eosid=2,
            padid=1,
            bosid=0,
            field_vocab=2756,
            n_positions=1024,
            n_ctx=1024,
            hidden_size=1024,
            position_vocab=31,
            pos_size=1024,
            field_size=1024,
            n_layer=12,
            n_head=16,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            activation_function="gelu",
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            max_len=85,
            top_k=4,
            top_p=0.9,
            use_coverage=True,
            use_copy_gate=True,
            copy_gate_penalty=0.7,
            coverage_penalty=0.02,
            dropout=0.2,
            deviceid=0

    ):

        self.vocab_size = vocab_size_or_config_json_file
        self.field_vocab = field_vocab
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.hidden_size = hidden_size
        self.position_vocab = position_vocab
        self.pos_size = pos_size
        self.field_size = field_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.activation_function = activation_function
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.max_length=max_len
        self.attention=True
        self.top_k = top_k
        self.top_p = top_p
        self.device =torch.device(("cuda:%d"% deviceid) if torch.cuda.is_available() else "cpu")
        self.eos=eosid
        self.pad=padid
        self.bos=bosid
        self.use_coverage = use_coverage
        self.use_copy_gate = use_copy_gate
        self.copy_gate_penalty = copy_gate_penalty
        self.coverage_penalty = coverage_penalty
        self.rand_unif_init_mag = 0.02
        self.trunc_norm_init_std = 1e-4
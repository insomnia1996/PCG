# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Parameter,init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math, random
from typing import *
from transformers import GPT2LMHeadModel, BartForConditionalGeneration, RobertaModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
from transformers.adapters import MAMConfig, PrefixTuningConfig


def init_lstm_wt(lstm, config):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear, config):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt, config):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt, config):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)




class SequenceLabeling(nn.Module):
    '''
    This content selector was modified from Ma's implementation  
    '''
    def __init__(self, 
                 config, wte=None,wfe=None,wpe=None) -> None:

        super().__init__()
        self.device = config.device
        self.pad=config.pad
        self.hidden_size = config.hidden_size
        self.field_size = config.field_size
        self.label_size = 2#key content or not.
        self.encoder = RobertaModel.from_pretrained("roberta-large")#roberta和bart共享同一个vocab 

        # 沿用decoder训练的embedding
        if wte:
            self.embedding = wte  # (input_dim, hidden_size)
        else:
            self.embedding=self.encoder.get_input_embeddings()
        if wfe:
            self.fembedding = wfe  # (input_dim, hidden_size)
        else:
            self.fembedding=nn.Embedding(config.field_vocab, config.field_size)
            init_wt_normal(self.fembedding.weight, config)
        if wpe:
            self.pembedding = wpe  # (input_dim, hidden_size)
        else:
            self.pembedding=nn.Embedding(config.position_vocab, config.pos_size)
            init_wt_normal(self.pembedding.weight, config)
        self.mlp=nn.Linear(config.hidden_size, self.label_size)
        self.dropout = nn.Dropout(0.3)

    def self_cross_entropy(self, input, target, ignore_index=None):
        '''自己用pytorch实现cross_entropy，
        有时候会因为各种原因，如：样本问题等，出现个别样本的loss为nan，影响模型的训练，
        不适用于所有样本loss都为nan的情况
        input:n*num_labels
        target:n
        '''
        input = input.contiguous().view(-1, input.shape[-1])
        log_prb = F.log_softmax(input, dim=1)

        one_hot = torch.zeros_like(input).scatter(1, target.view(-1, 1), 1)     # 将target转换成one-hot编码
        loss = -(one_hot * log_prb).sum(dim=1)                                  # n,得到每个样本的loss

        if ignore_index:                            # 忽略[PAD]的label
            non_pad_mask = target.ne(0)
            loss = loss.masked_select(non_pad_mask)
        
        not_nan_mask = ~torch.isnan(loss)           # 找到loss为非nan的样本
        loss = loss.masked_select(not_nan_mask).mean()
        return loss

    def _get_lengths(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x > 0).sum(-1)
        return lengths


    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                field: torch.Tensor,
                lpos: torch.Tensor,
                rpos:  torch.Tensor) -> Dict[str, torch.Tensor]:
        src=self.embedding(src)
        
        attn_mask=field.eq(0).to(self.device)
        field_pos = self.fembedding(field)+ self.pembedding(lpos)+ self.pembedding(rpos)
        out = self.encoder(inputs_embeds=src+field_pos, attention_mask=attn_mask).last_hidden_state
        out = self.dropout(out)
        out_logits = self.mlp(out)
        loss = self.self_cross_entropy(out_logits.view(-1, 2), tgt.view(-1))
        key_mask=torch.argmax(out_logits, dim=-1)
        #是不是分段训练效果更好？roberta没有做二分类的预训练任务，用小样本重新训感觉很困难？先跑着试试
        key_mask=~key_mask.eq(0)
        return key_mask*attn_mask, loss
    
        
    def predict(self, 
                src: Dict[str, torch.Tensor],
                field: Dict[str, torch.Tensor],
                lpos: Dict[str, torch.Tensor],
                rpos: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        with torch.no_grad(): 
            src=self.embedding(src)
            attn_mask=field.eq(0).to(self.device)
            field_pos = self.fembedding(field)+ self.pembedding(lpos)+ self.pembedding(rpos)
            out = self.encoder(src=src+field_pos, src_key_padding_mask=attn_mask)
            out_logits = self.mlp(out)
            key_mask=torch.argmax(out_logits, dim=-1)
            key_mask=~key_mask.eq(0)
            return key_mask*attn_mask


class FGateEncoder(nn.Module):
    '''re-define LSTM Encoder with field gating.'''
    def __init__(self, config, wte,wfe,wpe):
        self.device = config.device

        hidden_size = config.hidden_size
        field_size = config.field_size
        super(FGateEncoder, self).__init__()

        self.field_size = field_size
        self.hidden_size = hidden_size

        # normal LSTM
        self.w = Parameter(torch.Tensor( 4*hidden_size, 2*hidden_size)).to(self.device)
        #init.normal_(self.w)
        init_wt_unif(self.w, config)
        self.b = Parameter(torch.Tensor( 4*hidden_size, 1)).to(self.device)
        init_wt_unif(self.b, config)

        # field gate
        self.wf = Parameter(torch.Tensor( 2*hidden_size, hidden_size)).to(self.device)
        init_wt_unif(self.wf, config)
        self.bf = Parameter(torch.Tensor( 2*hidden_size, 1)).to(self.device)
        init_wt_unif(self.bf, config)

        #self.reset_weights()
        # 沿用decoder训练的embedding
        self.embedding = wte  # (input_dim, hidden_size)
        self.fembedding = wfe  # (field_dim, field_size)
        self.pembedding = wpe  # (pos_dim, pos_size)



    def reset_weights(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs, fields, pos, rpos, state=None):
        """Forward
        Args:
            inputs: [batch_size, seq_len]
            fields: [batch_size, seq_len] -- field encode number: 对field中每个单词，tokenize后embed，求均值作为femb。
            state: ([1, batch_size, hidden_size], [1, batch_size, hidden_size])
        """

        batch_size, seq_len= inputs.size()
        if state is None:
            h_t = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
            c_t = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        else:
            (h_t, c_t) = state
        inputs = self.embedding(inputs)# [batch_size, seq_len, hidden_size]
        fields = self.fembedding(fields)# [batch_size, seq_len, field_size]
        pos = self.pembedding(pos)# [batch_size, seq_len, pos_size]
        rpos = self.pembedding(rpos)# [batch_size, seq_len, pos_size]
        field_pos = fields + pos + rpos
        hidden_seq = []

        
        for t in range(seq_len):
            h_t = h_t.squeeze(0).t()#[hidden_size,batch_size]
            c_t = c_t.squeeze(0).t()  # [hidden_size,batch_size]
            x = inputs[:, t, :].t()#[hidden_size,batch_size]
            x = torch.cat((x,h_t), 0)#[2*hidden_size,batch_size]# concat attention
            fds = field_pos[:, t, :].t()  # [field_size,batch_size]
            # normal LSTM
            (i,f,o,c_hat)=torch.split(self.w @ x+self.b, self.hidden_size)#[4,hidden_size,batch_size]
            (l,z_hat) = torch.split(self.wf @ fds + self.bf, self.hidden_size)  # [2,hidden_size,batch_size]
            # input gate
            i = torch.sigmoid(i)#[hidden_size,batch_size]
            # forget gate
            f = torch.sigmoid(f)#[hidden_size,batch_size]
            # cell
            c_hat = torch.tanh(c_hat)#[hidden_size,batch_size]
            # output gate
            o = torch.sigmoid(o)#[hidden_size,batch_size]
            #field gate
            l = torch.sigmoid(l)#[hidden_size,batch_size]
            #field value
            z_hat = torch.tanh(z_hat)#[hidden_size,batch_size]

            c_next = f * c_t + i * c_hat + l * z_hat#[hidden_size,batch_size]
            h_next = o * torch.tanh(c_next)#[hidden_size,batch_size]
            c_t = c_next.t().unsqueeze(0).contiguous()# [1,batch_size, hidden_size]
            h_t = h_next.t().unsqueeze(0).contiguous()# [1,batch_size, hidden_size]
            hidden_seq.append(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)# [seq_len, batch_size, hidden_size]
        return hidden_seq, field_pos.transpose(0,1), (h_t, c_t)

class DualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dec_hidden = config.hidden_size
        self.field_size = config.field_size
        self.pos_size = config.pos_size
        self.device = config.device
        self.use_coverage = config.use_coverage
        if self.use_coverage:
            self.W_c = nn.Linear(1, 1, bias=True)
        self.attn = nn.Linear(self.hidden_size + self.dec_hidden, self.dec_hidden)
        self.attn2 = nn.Linear(self.hidden_size + self.dec_hidden, self.dec_hidden)
        self.v = nn.Linear(self.dec_hidden, 1, bias = False)
        self.v2 = nn.Linear(self.dec_hidden, 1, bias = False)
        self.w_gen = nn.Linear(self.hidden_size + 2 * self.dec_hidden, 1)
        
    def forward(self, in_t, s_t, encoder_outputs, field_pos, coverage):
        
        #s_t = [batch size, dec hid dim]
        #in_t = [batch size, dec hid dim]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        coverage = coverage.t()
        
        #repeat decoder hidden state src_len times
        st_rep = s_t.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        field_pos = field_pos.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]
        energy = torch.tanh(self.attn(torch.cat((st_rep, encoder_outputs), dim = 2))) 
        energy2 = torch.tanh(self.attn2(torch.cat((st_rep, field_pos), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]

        attention1 = self.v(energy).squeeze(2)
        attention2 = self.v2(energy2).squeeze(2)

        attn = attention1*attention2
        
        if self.use_coverage:
            
            coverage_penalty = torch.tanh(self.W_c(coverage.unsqueeze(-1)))  # [batch_size, enc_seq_len, hidden_dim]
            coverage_penalty = torch.exp(coverage_penalty - torch.max(coverage_penalty, -1, keepdim=True).values).squeeze(-1)
            final_attn = torch.div(attn * coverage_penalty, torch.sum(attn * coverage_penalty, -1, keepdim=True) + 1e-10)
        else:
            final_attn = torch.div(attn, torch.sum(attn, -1, keepdim=True) + 1e-10)

        context = torch.sum(encoder_outputs * torch.unsqueeze(final_attn, -1), 1, keepdim=False)  # sigma(ai*hi) [batch_size, hidden_size]
        
        # pointer generator
        # p_gen = torch.clamp(torch.sigmoid(context @ self.wc + torch.squeeze(st, 0) @ self.ws + in_t @ self.wx + self.b, min=1e-4, max=1-1e-4)) #[batch, 1]

        p_gen = torch.sigmoid(self.w_gen(torch.cat((context, s_t, in_t), dim=-1)))
        
        return final_attn, p_gen

class Decoder_baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.field_size = config.field_size
        self.max_length = config.max_length
        self.topk = config.top_k
        self.topp = config.top_p
        self.temperature = 1.0
        self.eos = config.eos
        self.device = config.device
        if config.attention:
            self.attention = DualAttention
        self.decoder = nn.LSTM(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout, inplace=True)
        self.ffn = nn.Sequential(
          nn.Linear(config.hidden_size , config.vocab_size),
          nn.Softmax(dim=-1))


        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)  # (input_dim, hidden_size)
        init_wt_normal(self.embedding.weight, config)
        self.fembedding = nn.Embedding(config.field_vocab, config.field_size)
        init_wt_normal(self.fembedding.weight, config)
        self.pembedding = nn.Embedding(config.position_vocab, config.pos_size)
        init_wt_normal(self.pembedding.weight, config)

        self.dual_attn = DualAttention(config)

    def forward(self, dec_input, fields, enc_outputs, enc_input, enc_state, dec_position):
        #dec_lstm hidden state st与enc_outputs & fields计算attention
        batch_size = dec_input.shape[-1]
        max_time = dec_input.shape[0]
        enc_seq_len = enc_outputs.shape[0]
        dec_input = dec_input.t()#[batch_size, ground_len]
        dec_position = dec_position.t()#[batch_size, ground_len]
        # coverage mechanism
        coverage_att_sum = torch.zeros([enc_seq_len, batch_size], dtype=torch.float32).to(self.device)
        coverloss = torch.zeros([batch_size, 1], dtype=torch.float32).to(self.device)

        # golden summary中copy原表内容的部分，copy标1，否则标0
        copy_gate_mask = torch.gt(dec_position,
                                  torch.zeros(dec_position.shape).to(self.device)).float()  # [batch_size, ground_len]
        # 补了右移位的遮罩
        copy_gate_mask = torch.cat((copy_gate_mask, torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)), dim=-1)  # [batch_size, ground_len+1]

        emit_ta = []  # output tensor array
        emit_gate = []  # copy_gate loss
        for t in range(self.max_length):
            if t==0:
                dec_in = self.embedding(torch.full((1,batch_size),self.eos,dtype=torch.int64).to(self.device))
                dec_in = self.dropout(dec_in)
                o_t, s_nt = self.decoder(dec_in, enc_state) #[1, batch_size, hidden_size]
            else:
                teacher_forcing=random.random()
                if teacher_forcing>0:#TODO
                    dec_in = self.embedding(dec_input[:,t-1]).unsqueeze(0) #[1, batch_size, embed_size]
                    dec_in = self.dropout(dec_in)
                else:
                    dec_in = self.embedding(dec_out.t()) #[1, batch_size, embed_size]
                    dec_in = self.dropout(dec_in)
                o_t, s_nt = self.decoder(dec_in, s_nt) #[1, batch_size, hidden_size]
            final_attn, p_gen = self.dual_attn(dec_in, o_t, fields, enc_outputs, coverage_att_sum)

            context = torch.sum(enc_outputs.transpose(0, 1) * torch.unsqueeze(final_attn, -1), 1, keepdim=False)
            gen_dist = self.ffn(context)
            # 用gpt2生成单词的logits
            att_dist = final_attn  # [batch_size, enc_seq_len], seq_len为encoder输入句子长度
            copy_dist = torch.zeros(gen_dist.shape, dtype=att_dist.dtype).to(self.device)  # [batch_size, vocab_size]
            copy_dist = copy_dist.scatter(dim=1, index=enc_input.t(), src=att_dist)  # [batch_size, vocab_size]
            copy_dist = torch.div(copy_dist, torch.sum(copy_dist, dim=-1, keepdim=True))
            # attn将表中出现的单词对应tokenid位置计算了attention score作为copy的分布，gpt输出作为generate的分布，依照概率将两种分布拟合。
            final_out_dist = p_gen * gen_dist + (1 - p_gen) * copy_dist  # [batch_size, vocab_size]; 最终输出词语分布
            dec_out = torch.argmax(final_out_dist, dim=-1, keepdim=True)# [batch_size, 1]
            copy_mask = copy_gate_mask[:,t].unsqueeze(1) #[batch_size, 1]
            emit_ta.append(final_out_dist.unsqueeze(1))#[batch_size, 1, vocab_size]
            emit_gate.append(p_gen * copy_mask) #[batch_size, 1]

            #add coverloss
            this_coverloss = torch.sum(torch.minimum(coverage_att_sum.t(), final_attn), dim=-1)#[batch_size, enc_seq_len]
            coverloss = coverloss + this_coverloss
            coverage_att_sum = coverage_att_sum + final_attn.transpose(0,1)
            # TODO: reconsider stop condition
            finished = t >= max_time  # finish state
            if finished:
                #print("Time %d on finish state." % t)
                break

        #output: log softmax logits
        emit_ta = torch.cat(emit_ta,dim=1)#[batch_size, ground_len+1, vocab_size]
        emit_gate = torch.cat(emit_gate, dim=1)#[batch_size, ground_len+1], reduce_sum即可得到batch_copy_loss, shape:[batch_size, 1]
        return emit_ta, coverloss, emit_gate


class Decoder_nlg(nn.Module):
    # 先输入[0:t-1]时刻的ground truth，通过GPT-2生成第t时刻的单词作为生成结果输入到dual attention做auto regression。
    # 再考虑copy结果，最后综合起来。
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.field_size = config.field_size
        self.pos_size = config.pos_size
        self.max_length = config.max_length
        self.topk = config.top_k
        self.topp = config.top_p
        self.temperature = 0.9
        self.eos = config.eos
        self.device = config.device
        self.use_coverage = config.use_coverage
        self.step_gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        #self.step_gpt.resize_token_embeddings(len(tokenizer))
        self.embedding = self.step_gpt.transformer.wte  # (input_dim, hidden_size)
        #self.embedding.weight.requires_grad = False
        self.fembedding = nn.Embedding(config.field_vocab, config.field_size)
        init_wt_normal(self.fembedding.weight, config)

        # cond 1: self-trained embedding
        self.pembedding = nn.Embedding(config.position_vocab, config.pos_size)
        init_wt_normal(self.pembedding.weight, config)
        # cond2: wpe fixed embedding
        # self.pembedding = self.step_gpt.transformer.wpe
        # self.pembedding.weight.requires_grad = False
        self.dual_attn = DualAttention(config)

    def forward(self, dec_input, fields, enc_outputs, enc_input, ground_truth, ground_len, summary_fields, summary_pos_input,
                summary_rpos_input):
        '''
        enc_input用来生成copy_logits
        ground_truth, summary_pos_input都是用作训练时计算loss
        '''
        # step_gpt第一步用表格格式化输入作为past。之后时间步都用golden summary做teacher forcing.
        # dec_input(其实是context_in):训练时用表格模板作为初始输入(已tokenize为tensor)。
        # dec_input:[dec_seq_len, batch_size]; 表格模板化输出：title : once an eagle, author : anton myrer, country : united states, language : english, genre : war, publisher : holt, rinehart, and winston, publication date : 1968, media type : print ( hardback paperback ), pages : 1312, isbn :.,.Book description:

        # st: 训练时ground truth 过GPT2的hidden state, gen hidden_state
        # st:[1, batch_size, hidden_size]; results['hidden'][:-1](训练时直接输入ground_truth，计算出所有的hidden的[-1]，再concat起来作为全部的hidden)
        # fields & enc_outputs只用来在dualattn模块算相似度。
        # fields:[enc_seq_len, batch_size, field_size +2 * pos_size]
        # enc_outputs:[enc_seq_len, batch_size, hidden_size]
        # enc_input:[enc_seq_len, batch_size];  表格value输出：once an eagle anton myrer united states english war holt, rinehart, and winston 1968 print ( hardback paperback ) 1312.
        # enc_input用来生成复制概率下的logits
        # ground_truth:[ground_len, batch_size]; 表格内容总结：once an eagle is a 1968 war novel by american author anton myrer.
        # summary_fields:[ground_len, batch_size]
        # summary_pos_input:[ground_len, batch_size]
        # summary_rpos_input:[ground_len, batch_size]
        batch_size = dec_input.shape[-1]
        enc_seq_len = enc_outputs.shape[0]
        past = None

        ground_truth = ground_truth.t()  # [batch_size, ground_len]
        summary_fields = summary_fields.t()  # [batch_size, ground_len]
        summary_fields = self.fembedding(summary_fields)  # [batch_size, ground_len, field_size]
        summary_pos_input = summary_pos_input.t()  # [batch_size, ground_len]
        summary_position = summary_pos_input
        summary_pos_input = self.pembedding(summary_pos_input)  # [batch_size, ground_len, pos_size]
        summary_rpos_input = summary_rpos_input.t()  # [batch_size, ground_len]
        summary_rpos_input = self.pembedding(summary_rpos_input)  # [batch_size, ground_len, pos_size]
        
        # coverage mechanism
        if self.use_coverage:
            coverage_att_sum = torch.zeros([enc_seq_len, batch_size], dtype=torch.float32).to(self.device)
            coverloss = []
        else:
            coverage_att_sum=None

        emit_ta = []  # output tensor array
        emit_gate = []  # copy_gate loss
        #golden mask
        max_declen = max(ground_len)
        batch_num = [[1]*dec_len + [0]*(max_declen-dec_len) for dec_len in ground_len]
        ground_mask = torch.tensor(batch_num, dtype=torch.float32).to(self.device)
        # golden summary中copy原表内容的部分，copy标1，否则标0
        copy_gate_mask = torch.gt(summary_position,
                                  torch.zeros(summary_position.shape).to(self.device)).float()  # [batch_size, ground_len]
        # 补了右移位的遮罩
        copy_gate_mask = torch.cat((copy_gate_mask, torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)),
                                   dim=-1)  # [batch_size, ground_len+1]

        for t in range(min(ground_truth.size(-1), self.max_length + 1)):  # max_length加了EOS
            # 初始化时已经做过一次step_gpt()得到x0

            if t == 0:
                dec_in = dec_input.t()  # [batch_size,context_len]，t=0时刻预测输出golden_summary[0]，同时golden_summary[0]作为t=1时输入。
                dec_mask = torch.ones_like(dec_in, dtype=torch.float32).to(self.device)
            else:
                dec_in = ground_truth[:, t].unsqueeze(-1)  # [batch_size,1] summary在t时刻输入t时刻单词(跳过BOS)。（shift_right） teacher forcing
                dec_mask= ground_mask[:, t].unsqueeze(-1)  # [batch_size,1]
            output = self.step_gpt(inputs_embeds=self.embedding(dec_in),
                                   attention_mask=dec_mask, 
                                   past_key_values=past,output_hidden_states=True)  # 看transformers generate的earlystop怎么做的
            gen_dist = output.logits[:, -1, :] # [batch_size, vocab_size]
            #对lmhead输出加了softmax，与copydist相匹配
            gen_dist = F.softmax(gen_dist, dim=-1)
            past = output.past_key_values  # (num_layers,2,batch_size ,num_heads,seq_len,hidden_size), 12=num_layers, 2=(k,v)为元组，需要past[0][0].shape获取shape
            hidden_states = output.hidden_states  # 13*(batch_size, out_seq_len, hidden_size=768). 13=one for the output of the embeddings + one for the output of each layer.

            # 生成出的既可能是field也可能是value,分别进行attn
            token = torch.argmax(gen_dist, dim=-1, keepdim=False)  # [batch_size,]
            # dec_in to dec_in_emb & dec_in_field
            s_t = hidden_states[-1][:, -1, :]  # [batch_size,hidden_size], top layer & seq[-1]
            # in_t就是t时刻ground truth输入的embedding表示（+fpemb），既作为GPT输入，也作为dualattn的copy输入。
            if t == 0:
                # GPT生成的作为gen_dist，并将gpt2预测输出作为<BOS>输入dualattn。
                in_t = self.embedding(dec_in[:, -1])  # [batch_size, hidden_size]
                field_pos = torch.zeros(
                    (batch_size, self.hidden_size)).to(self.device)  # [batch_size, hidden_size]
            else:
                # GPT生成的作为gen_dist，并将ground truth 输入dualattn。
                # decoder_g 要修改这个dec_in为sampling结果
                in_t = self.embedding(dec_in[:, -1])  # [batch_size, hidden_size]

                summary_fields_t = summary_fields[:, t - 1, :]  # [batch_size, field_size]
                pos_t = summary_pos_input[:, t - 1, :]  # [batch_size, pos_size]
                rpos_t = summary_rpos_input[:, t - 1, :]  # [batch_size, pos_size]
                field_pos = summary_fields_t + pos_t + rpos_t  # [batch_size, hidden_size]
            in_t = in_t# [batch_size, hidden_size]
            # pass the hidden weights(s_t) into the attention layer to get gen gate probability
            # final_attn: [batch_size, enc_seq_len]. p_gen: [batch_size, 1]
            final_attn, p_gen = self.dual_attn(in_t, s_t, enc_outputs, fields, coverage_att_sum)

            # 用gpt2生成单词的logits
            att_dist = final_attn  # [batch_size, enc_seq_len], seq_len为encoder输入句子长度
            copy_dist = torch.zeros(gen_dist.shape, dtype=att_dist.dtype).to(self.device)  # [batch_size, vocab_size]
            copy_dist = copy_dist.scatter(dim=1, index=enc_input.t(), src=att_dist)  # [batch_size, vocab_size]
            #copy_dist = torch.div(copy_dist, torch.sum(copy_dist, dim=-1, keepdim=True))
            # attn将表中出现的单词对应tokenid位置计算了attention score作为copy的分布，gpt输出作为generate的分布，依照概率将两种分布拟合。
            final_out_dist = p_gen * gen_dist + (1 - p_gen) * copy_dist  # [batch_size, vocab_size]; 最终输出词语分布
            final_out_dist = final_out_dist.clamp(min=1e-12)
            # write to tensor array
            copy_mask = copy_gate_mask[:, t].unsqueeze(1)  # [batch_size, 1]
            
            
            emit_ta.append(final_out_dist.unsqueeze(1))  # [batch_size, 1, vocab_size]
            emit_gate.append(p_gen * copy_mask)  # [batch_size, 1]

            # add coverloss
            if self.use_coverage:
                this_coverloss = torch.sum(torch.minimum(coverage_att_sum.t(), final_attn),
                                    dim=-1, keepdim=True)  # [batch_size, 1]
                coverloss.append(this_coverloss)
                coverage_att_sum = coverage_att_sum + final_attn.transpose(0, 1)
        emit_ta = torch.cat(emit_ta, dim=1)  # [batch_size, ground_len, vocab_size]
        emit_gate = torch.cat(emit_gate, dim=1)  # [batch_size, ground_len], reduce_sum即可得到batch_copy_loss, shape:[batch_size, 1]
        if self.use_coverage:
            coverloss = torch.cat(coverloss, dim=1)# [batch_size, ground_len]
            return emit_ta, coverloss, emit_gate
        else:
            return emit_ta, torch.zeros_like(emit_gate), emit_gate

    def regularization(self, tensor):
        if len(tensor.shape) > 2:
            torch.unsqueeze(tensor, -1)
        # tensor: [batch_size,seq_len]
        tensor = torch.exp(tensor - torch.max(tensor, -1, keepdim=True).values)
        tensor = torch.div(tensor, torch.sum(tensor, -1, keepdim=True) + 1e-10)
        # output tensor: [batch_size,seq_len]
        return tensor
    
    def greedy_decode(self, context_in, field_pos, enc_outputs, enc_input):
        past=None
        context_in = context_in.t()
        batch_size = enc_input.shape[-1]
        enc_seq_len = enc_outputs.shape[0]
        # coverage mechanism
        if self.use_coverage:
            coverage_att_sum = torch.zeros([enc_seq_len, batch_size], dtype=torch.float32).to(self.device)
        else:
            coverage_att_sum = None
        output_tokens = []

        for t in range(self.max_length):  # 需要添加finished参数#
            if t==0:
                dec_in = context_in #[1, context_len]
            else:
                dec_in = x_t # [1, 1]
            dec_embd = self.embedding(dec_in)# [1, input_len, hidden_size]
            output = self.step_gpt(inputs_embeds=dec_embd, output_hidden_states=True, past_key_values=past) 
            gen_dist = output.logits[:, -1, :]  # [1, vocab_size]
            #对lmhead输出加了softmax，与copydist相匹配
            gen_dist = F.softmax(gen_dist, dim=-1)
            hidden_states = output.hidden_states
            past = output.past_key_values
            s_t = hidden_states[-1][:, -1, :] # 1 x hidden_size
            
            # in_t就是t时刻ground truth输入的embedding表示（+fpemb），既作为BART输入，也作为dualattn的copy输入。
            final_attn, p_gen = self.dual_attn(dec_embd[:,-1,:], s_t, enc_outputs, field_pos, coverage_att_sum)
            
            # 用gpt2生成单词的logits
            att_dist = final_attn  # [1, enc_seq_len], seq_len为encoder输入句子长度
            copy_dist = torch.zeros(gen_dist.shape, dtype=att_dist.dtype).to(self.device)  # [1, vocab_size]
            copy_dist = copy_dist.scatter(dim=1, index=enc_input.t(), src=att_dist)  # [1, vocab_size]
            
            #copy_dist = torch.div(copy_dist, torch.sum(copy_dist, dim=-1, keepdim=True))

            # attn将表中出现的单词对应tokenid位置计算了attention score作为copy的分布，gpt输出作为generate的分布，依照概率将两种分布拟合。
            final_out_dist = p_gen * gen_dist + (1 - p_gen) * copy_dist  # [1, vocab_size]; 最终输出词语分布
            final_out_dist = final_out_dist.clamp(min=1e-12)
            x_t = torch.argmax(final_out_dist, dim=-1, keepdim=True)
            output_tokens.append(x_t)
            if torch.sum(x_t == self.eos) == batch_size:# all sentences meet eos_token 
                break

        output_tokens = torch.cat(output_tokens, dim=1)  # [1, summary_len]
        return output_tokens


class Decoder_prompt(nn.Module):
    # 先输入[0:t-1]时刻的ground truth，通过GPT-2生成第t时刻的单词作为生成结果输入到dual attention做auto regression。
    # 再考虑copy结果，最后综合起来。
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.field_size = config.field_size
        self.pos_size = config.pos_size
        self.max_length = config.max_length
        self.topk = config.top_k
        self.topp = config.top_p
        self.temperature = 0.9
        self.eos = config.eos
        self.bos=config.bos
        self.pad=config.pad
        self.device = config.device
        self.use_coverage = config.use_coverage
        self.step_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        
        self.bart_model = self.step_bart.model
        ptuning_config = PrefixTuningConfig(cross_prefix=False, leave_out=[12,13,14,15,16,17,18,19,20,21,22,23])
        adapter_config = MAMConfig(prefix_tuning=ptuning_config)
        
        self.bart_model.add_adapter("mamconfig", config=adapter_config)
        self.bart_model.train_adapter("mamconfig")
        self.bart_dec = self.step_bart.get_decoder()
        self.lm_head = self.step_bart.lm_head
        self.embedding = self.bart_model.shared
        self.fembedding = nn.Embedding(config.field_vocab, config.field_size)
        init_wt_normal(self.fembedding.weight, config)

        # cond 1: self-trained embedding
        self.pembedding = nn.Embedding(config.position_vocab, config.pos_size)
        init_wt_normal(self.pembedding.weight, config)
        # cond2: wpe fixed embedding
        # self.pembedding = self.step_gpt.transformer.wpe
        # self.pembedding.weight.requires_grad = False
        self.dual_attn = DualAttention(config)

    def forward(self, cont_in, fields, enc_outputs, enc_input, ground_truth, ground_len, summary_fields, summary_pos_input,
                summary_rpos_input):
        '''
        enc_input用来生成copy_logits
        ground_truth, summary_pos_input都是用作训练时计算loss
        '''
        # step_gpt第一步用表格格式化输入作为past。之后时间步都用golden summary做teacher forcing.
        # cont_in(其实是context_in):训练时用表格模板作为初始输入(已tokenize为tensor)。
        # cont_in:[dec_seq_len, batch_size]; 表格模板化输出：title : once an eagle, author : anton myrer, country : united states, language : english, genre : war, publisher : holt, rinehart, and winston, publication date : 1968, media type : print ( hardback paperback ), pages : 1312, isbn :.,.Book description:

        # st: 训练时ground truth 过GPT2的hidden state, gen hidden_state
        # st:[1, batch_size, hidden_size]; results['hidden'][:-1](训练时直接输入ground_truth，计算出所有的hidden的[-1]，再concat起来作为全部的hidden)
        # fields & enc_outputs只用来在dualattn模块算相似度。
        # fields:[enc_seq_len, batch_size, field_size +2 * pos_size]
        # enc_outputs:[enc_seq_len, batch_size, hidden_size]
        # enc_input:[enc_seq_len, batch_size];  表格value输出：once an eagle anton myrer united states english war holt, rinehart, and winston 1968 print ( hardback paperback ) 1312.
        # enc_input用来生成复制概率下的logits
        # ground_truth:[ground_len, batch_size]; 表格内容总结：once an eagle is a 1968 war novel by american author anton myrer.
        # summary_fields:[ground_len, batch_size]
        # summary_pos_input:[ground_len, batch_size]
        # summary_rpos_input:[ground_len, batch_size]
        batch_size = cont_in.shape[-1]
        enc_seq_len = enc_outputs.shape[0]
        past = None

        ground_truth = ground_truth.t()  # [batch_size, ground_len]
        summary_pos_input = summary_pos_input.t()  # [batch_size, ground_len]
        summary_position = summary_pos_input

        
        # coverage mechanism
        if self.use_coverage:
            coverage_att_sum = torch.zeros([enc_seq_len, batch_size], dtype=torch.float32).to(self.device)
            coverloss = []
        else:
            coverage_att_sum=None

        emit_ta = []  # output tensor array
        emit_gate = []  # copy_gate loss

        # golden summary中copy原表内容的部分，copy标1，否则标0
        copy_gate_mask = torch.gt(summary_position,
                                  torch.zeros(summary_position.shape).to(self.device)).float()  # [batch_size, ground_len]
        # 补了右移位的遮罩
        copy_gate_mask = torch.cat((copy_gate_mask, torch.zeros((batch_size, 1), dtype=torch.float32).to(self.device)),
                                   dim=-1)  # [batch_size, ground_len+1]
        #bart encoder 
        cont_in=cont_in.t()

        bartenc=self.bart_model(input_ids=cont_in, output_hidden_states=True).encoder_last_hidden_state
        
        for t in range(min(ground_truth.size(-1), self.max_length + 1)):  # max_length加了EOS
            token_t = ground_truth[:,t].unsqueeze(-1)
            output = self.bart_dec(inputs_embeds=self.embedding(token_t),
                                   encoder_hidden_states=bartenc,
                                   past_key_values=past,
                                   output_hidden_states=True)
            s_t = output.hidden_states[-1][:,-1,:]# [batch_size, hidden_size]
            past = output.past_key_values
            gen_dist = self.lm_head(s_t)
            #对lmhead输出加了softmax，与copydist相匹配
            gen_dist = F.softmax(gen_dist, dim=-1)
            in_t = self.embedding(token_t[:,-1])# [batch_size, hidden_size]
            # pass the hidden weights(s_t) into the attention layer to get gen gate probability
            # final_attn: [batch_size, enc_seq_len]. p_gen: [batch_size, 1]
            final_attn, p_gen = self.dual_attn(in_t, s_t, enc_outputs, fields, coverage_att_sum)

            # 用gpt2生成单词的logits
            att_dist = final_attn  # [batch_size, enc_seq_len], seq_len为encoder输入句子长度
            copy_dist = torch.zeros(gen_dist.shape, dtype=att_dist.dtype).to(self.device)  # [batch_size, vocab_size]
            copy_dist = copy_dist.scatter(dim=1, index=enc_input.t(), src=att_dist)  # [batch_size, vocab_size]
            #copy_dist = torch.div(copy_dist, torch.sum(copy_dist, dim=-1, keepdim=True))
            # attn将表中出现的单词对应tokenid位置计算了attention score作为copy的分布，gpt输出作为generate的分布，依照概率将两种分布拟合。
            final_out_dist = p_gen * gen_dist + (1 - p_gen) * copy_dist  # [batch_size, vocab_size]; 最终输出词语分布
            final_out_dist = final_out_dist.clamp(min=1e-12)
            # write to tensor array
            copy_mask = copy_gate_mask[:, t].unsqueeze(1)  # [batch_size, 1]
            
            
            emit_ta.append(final_out_dist.unsqueeze(1))  # [batch_size, 1, vocab_size]
            emit_gate.append(p_gen * copy_mask)  # [batch_size, 1]

            # add coverloss
            if self.use_coverage:
                this_coverloss = torch.sum(torch.minimum(coverage_att_sum.t(), final_attn),
                                    dim=-1, keepdim=True)  # [batch_size, 1]
                coverloss.append(this_coverloss)
                coverage_att_sum = coverage_att_sum + final_attn.transpose(0, 1)
        emit_ta = torch.cat(emit_ta, dim=1)  # [batch_size, ground_len, vocab_size]
        emit_gate = torch.cat(emit_gate, dim=1)  # [batch_size, ground_len], reduce_sum即可得到batch_copy_loss, shape:[batch_size, 1]
        if self.use_coverage:
            coverloss = torch.cat(coverloss, dim=1)# [batch_size, ground_len]
            return emit_ta, coverloss, emit_gate
        else:
            return emit_ta, torch.zeros_like(emit_gate), emit_gate

    def regularization(self, tensor):
        if len(tensor.shape) > 2:
            torch.unsqueeze(tensor, -1)
        # tensor: [batch_size,seq_len]
        tensor = torch.exp(tensor - torch.max(tensor, -1, keepdim=True).values)
        tensor = torch.div(tensor, torch.sum(tensor, -1, keepdim=True) + 1e-10)
        # output tensor: [batch_size,seq_len]
        return tensor
    
    def greedy_decode(self, context_in, field_pos, enc_outputs, enc_input):
        past=None
        batch_size = enc_input.shape[-1]
        enc_seq_len = enc_outputs.shape[0]
        # coverage mechanism
        if self.use_coverage:
            coverage_att_sum = torch.zeros([enc_seq_len, batch_size], dtype=torch.float32).to(self.device)
        else:
            coverage_att_sum = None
        output_tokens = []
        #bart encoder 
        cont_in=context_in.t()
        context_mask = torch.ones_like(cont_in)
        context_mask = context_mask.masked_fill(cont_in.eq(self.pad), 0.0).type(torch.FloatTensor).to(self.device)

        bartenc=self.bart_model(input_ids=cont_in, output_hidden_states=True).encoder_last_hidden_state

        for t in range(self.max_length):  # 需要添加finished参数#
            if t==0:
                dec_in = torch.full((batch_size, 1),self.bos,dtype=torch.int64).to(self.device)
            else:
                dec_in = x_t # [1, 1]
            dec_embd = self.embedding(dec_in)# [1, input_len, hidden_size]
            output = self.bart_dec(inputs_embeds=dec_embd,
                                   encoder_hidden_states=bartenc,
                                   encoder_attention_mask=context_mask,
                                   past_key_values=past,output_hidden_states=True)  # 看transformers generate的earlystop怎么做的
            s_t = output.hidden_states[-1][:,-1,:]# [batch_size, hidden_size]
            past = output.past_key_values
            gen_dist = self.lm_head(s_t)
            #对lmhead输出加了softmax，与copydist相匹配
            gen_dist = F.softmax(gen_dist, dim=-1)
            in_t = self.embedding(dec_in[:,-1])# [batch_size, hidden_size]
            # pass the hidden weights(s_t) into the attention layer to get gen gate probability
            # final_attn: [batch_size, enc_seq_len]. p_gen: [batch_size, 1]
            final_attn, p_gen = self.dual_attn(in_t, s_t, enc_outputs, field_pos, coverage_att_sum)
            
            # 用gpt2生成单词的logits
            att_dist = final_attn  # [1, enc_seq_len], seq_len为encoder输入句子长度
            copy_dist = torch.zeros(gen_dist.shape, dtype=att_dist.dtype).to(self.device)  # [1, vocab_size]
            copy_dist = copy_dist.scatter(dim=1, index=enc_input.t(), src=att_dist)  # [1, vocab_size]
            
            #copy_dist = torch.div(copy_dist, torch.sum(copy_dist, dim=-1, keepdim=True))

            # attn将表中出现的单词对应tokenid位置计算了attention score作为copy的分布，gpt输出作为generate的分布，依照概率将两种分布拟合。
            final_out_dist = p_gen * gen_dist + (1 - p_gen) * copy_dist  # [1, vocab_size]; 最终输出词语分布
            final_out_dist = final_out_dist.clamp(min=1e-12)
            x_t = torch.argmax(final_out_dist, dim=-1, keepdim=True)
            output_tokens.append(x_t)
            if torch.sum(x_t == self.eos) == batch_size:# all sentences meet eos_token 
                break

        output_tokens = torch.cat(output_tokens, dim=1)  # [1, summary_len]
        return output_tokens

import torch
from torch import nn
from dynamic_crf_layer import DynamicCRF
from torch.nn import CrossEntropyLoss

train_fct = CrossEntropyLoss()
class TopLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, crf_low_rank, crf_beam_size, padding_idx):
        super(TopLayer, self).__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.crf_layer = DynamicCRF(num_embedding = vocab_size, low_rank = crf_low_rank, 
                                    beam_size = crf_beam_size)

        self.one_more_layer_norm = nn.LayerNorm(embed_dim)
        self.tgt_word_prj = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, src_representation, tgt_input):
        '''
            src_representation: bsz x seqlen x embed_dim
            tgt_input: bsz x seqlen
        '''
        bsz, seqlen = tgt_input.size()
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)
        # compute mle loss
        logits = emissions.transpose(0,1).contiguous()
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        labels = tgt_input.clone()
        labels[labels[:, :] == self.padding_idx] = -100
        mle_loss = train_fct(logits.view(-1, self.vocab_size), labels.view(-1)) # averaged mle loss

        # compute crf loss
        emissions = emissions.transpose(0, 1) # [bsz x src_len x vocab_size]

        emission_mask = ~tgt_input.eq(self.padding_idx) # [bsz x src_len]
        batch_crf_loss = -1 * self.crf_layer(emissions, tgt_input, emission_mask) # [bsz]
        assert batch_crf_loss.size() == torch.Size([bsz])
        # create tgt mask
        tgt_mask = torch.ones_like(tgt_input)
        tgt_mask = tgt_mask.masked_fill(tgt_input.eq(self.padding_idx), 0.0).type(torch.FloatTensor)
        if tgt_input.is_cuda:
            tgt_mask = tgt_mask.cuda(tgt_input.get_device())
        crf_loss = torch.sum(batch_crf_loss) / torch.sum(tgt_mask)
        return mle_loss, crf_loss

    def decoding(self, src_representation):
        bsz, seqlen, _ = src_representation.size()
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        _, finalized_tokens = self.crf_layer.forward_decoder(emissions)
        assert finalized_tokens.size() == torch.Size([bsz, seqlen])
        return finalized_tokens

    def selective_decoding(self, src_representation, selective_mask):
        bsz, seqlen, _ = src_representation.size()
        src_representation = src_representation.transpose(0, 1) # seqlen x bsz x embed_dim
        src = src_representation

        emissions = self.tgt_word_prj(src.contiguous().view(-1, self.embed_dim)).view(seqlen, bsz, self.vocab_size)

        emissions = emissions.transpose(0, 1) # [bsz, seqlen, vocab_size]
        assert emissions.size() == selective_mask.size()
        emissions = emissions + selective_mask # mask the impossible token set

        _, finalized_tokens = self.crf_layer.forward_decoder(emissions)
        assert finalized_tokens.size() == torch.Size([bsz, seqlen])
        return finalized_tokens        


class ContentPlanner(nn.Module):
    def __init__(self, model_name, tkn, crf_low_rank=32, crf_beam_size=128, hidden_size=256):
        super(ContentPlanner, self).__init__()
        print("model name: ", model_name)
        from transformers import RobertaModel
        self.tokenizer = tkn
        self.embed_model = RobertaModel.from_pretrained(model_name)
        self.word_embeds = self.embed_model.get_input_embeddings()
        self.hidden_size = hidden_size
        self.lamda = 0.7
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.word_embeds.embedding_dim
        self.model = nn.LSTM(self.embed_dim, self.hidden_size // 2,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.pad_token_id = 1
        self.target_vocab_size = 53
        self.toplayer = TopLayer(self.target_vocab_size, self.hidden_size, crf_low_rank, 
            crf_beam_size, 52)#totally 52 length and 1 pad_token_id

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)

        parameter_path = ckpt_save_path + '/parameters/'
        if os.path.exists(parameter_path):
            pass
        else: # recursively construct directory
            os.makedirs(parameter_path, exist_ok=True)

        torch.save({'model':self.state_dict()}, 
            parameter_path + r'model.bin')

    def load_pretrained_model(self, ckpt_save_path):
        print ('Loading pre-trained parameters...')
        parameter_path = ckpt_save_path + '/parameters/model.bin'
        if torch.cuda.is_available():
            print ('Cuda is available.')
            model_ckpt = torch.load(parameter_path)
        else:
            print ('Cuda is not available.')
            model_ckpt = torch.load(parameter_path, map_location='cpu')
        model_parameters = model_ckpt['model']
        self.load_state_dict(model_parameters)

    def prepare_input(self, field_in, text_in):
        assert len(field_in)==len(text_in)
        assert len(field_in)==1 #batch_size can only be 1
        k,v = field_in[0], text_in[0]
        assert len(k)==len(v)
        sent=[]
        self.hidden=(torch.randn(2, 1, self.hidden_size // 2).to(self.device),
                     torch.randn(2, 1, self.hidden_size // 2).to(self.device))
        for idx in range(len(k)):
            kemb = torch.sum(self.word_embeds(torch.tensor(k[idx], dtype=torch.int64).to(self.device)), dim=0, keepdim=True)
            vemb = torch.sum(self.word_embeds(torch.tensor(v[idx], dtype=torch.int64).to(self.device)), dim=0, keepdim=True)
            sent.append(self.lamda*kemb+(1-self.lamda)*vemb)
        #sent shape: table_len, emb_size
        sent = torch.cat(sent, dim=0)#tablen, emb_size
        sent = sent.unsqueeze(0)
        return sent#(1, tablen, emb_size)

    def forward(self, field_in, text_in, tgt_input):
        '''
            field_in: bsz x tablen x keylen
            tgt_input: bsz x seqlen
        '''
        bsz, seqlen = len(field_in), len(field_in[0])
        assert bsz==1
        sent_in = self.prepare_input(field_in, text_in)#bsz x tablen x emb_size
        outputs, self.hidden = self.model(sent_in, self.hidden)
        src_representation = outputs
        assert src_representation.size() == torch.Size([bsz, seqlen, self.hidden_size])
        mle_loss, crf_loss = self.toplayer(src_representation, tgt_input)
        return mle_loss, crf_loss

    def parse_one_output(self, id_list, field_in):
        path=[]
        #DONE: 这里应该把已经出现的重复token去掉
        #去重以后效果明显好起来了，可以先按这样计算。
        #TODO: 修改predict代码，变成2step model，先用预测结果生成test文本的content plan。
        short_list = []
        for idx in id_list:
            if idx == 0 or idx>=len(field_in) or idx in path:
                continue
            else:
                short_list.extend(field_in[idx]+[6])
                path.append(idx)
        if len(short_list) == 0:
            short_list = [15483]
        elif short_list[-1]==6:
            short_list[-1]=15483
        #key1,key2,key3| ...context...
        result = self.tokenizer.decode(short_list)
        return result

    def parse_batch_output(self, finalized_tokens, field_in):
        predictions = finalized_tokens.detach().cpu().tolist()
        result = []
        for idx, item in enumerate(predictions):
            one_res = self.parse_one_output(item, field_in[idx])
            result.append(one_res)
        print("parsing results: ", result)
        return result

    def decode(self, field_in, text_in):
        bsz, seqlen = len(field_in), len(field_in[0])
        assert bsz==1
        sent_in = self.prepare_input(field_in, text_in)
        outputs, self.hidden = self.model(sent_in, self.hidden)
        src_representation = outputs
        finalized_tokens = self.toplayer.decoding(src_representation)
        print("decoded tokens:", finalized_tokens)

        return self.parse_batch_output(finalized_tokens, field_in)

    # the part of selective decoding
    def produce_selective_mask(self, bsz, seqlen, vocab_size, selective_id_list):
        assert len(selective_id_list) == bsz
        res_list = []
        for idx in range(bsz):
            one_selective_id_list = selective_id_list[idx]
            one_tensor = torch.ones(vocab_size) * float('-inf')
            for s_id in one_selective_id_list:
                one_tensor[s_id] = 0.
            one_res = [one_tensor for _ in range(seqlen)]
            one_res = torch.stack(one_res, dim=0)
            assert one_res.size() == torch.Size([seqlen, vocab_size])
            res_list.append(one_res)
        res_mask = torch.stack(res_list, dim = 0)
        assert res_mask.size() == torch.Size([bsz, seqlen, vocab_size])
        return res_mask

    def selective_decoding(self, src_input, selective_id_list):
        '''
            selective_id_list: 
                A list of length bsz. Each item contains the selective ids of content plan. 
                The final generated path should be formatted with these selective ids. 
        '''
        bsz, seqlen = src_input.size()
        selective_mask = self.produce_selective_mask(bsz, seqlen, self.target_vocab_size, selective_id_list)
        if src_input.is_cuda:
            selective_mask = selective_mask.cuda(src_input.get_device())

        # create src mask matrix
        src_mask = torch.ones_like(src_input)
        src_mask = src_mask.masked_fill(src_input.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        if src_input.is_cuda:
            src_mask = src_mask.cuda(src_input.get_device())

        outputs = self.model(input_ids=src_input, attention_mask=src_mask)
        src_representation = outputs[0]
        finalized_tokens = self.toplayer.selective_decoding(src_representation, selective_mask)
        return self.parse_batch_output(finalized_tokens)
    @property
    def device(self):
        return self.word_embeds.weight.device

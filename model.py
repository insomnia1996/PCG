# -*- coding: utf-8 -*-
import torch
from layer import FGateEncoder,Decoder_nlg, Decoder_prompt, SequenceLabeling
from transformers import BartForConditionalGeneration
#T5会根据labels自动生成decoder_input_ids，不用传
import torch.nn as nn
import torch.nn.functional as F
from transformers.adapters import MAMConfig, PrefixTuningConfig
import encoder


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.device = config.device
        self.decoder = Decoder_nlg(config).to(self.device)
        #self.gen = Decoder_for_test(config).to(device)
        self.encoder = FGateEncoder(config, self.decoder.embedding, self.decoder.fembedding,self.decoder.pembedding).to(self.device)
        self.config = config
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.use_coverage = config.use_coverage
        self.use_copy_gate = config.use_copy_gate
        self.copy_gate_penalty = config.copy_gate_penalty
        self.coverage_penalty = config.coverage_penalty
        #self.basic_loss = nn.CrossEntropyLoss(reduction='none')
        self.basic_loss = nn.NLLLoss(reduction='none')



    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference
            dec_in: gold truth
            dec_out: gold truth + eos*1
        Returns:
            feed_dict
        """

        self.encoder_input = torch.tensor(x['enc_in'], dtype=torch.int64).to(self.device)
        self.encoder_field = torch.tensor(x['enc_fd'], dtype=torch.int64).to(self.device)
        self.encoder_len = torch.tensor(x['enc_len'], dtype=torch.int64).to(self.device)
        self.encoder_pos = torch.tensor(x['enc_pos'], dtype=torch.int64).to(self.device)
        self.encoder_rpos = torch.tensor(x['enc_rpos'], dtype=torch.int64).to(self.device)
        self.context = torch.tensor(x['context'], dtype=torch.int64).to(self.device)
        self.gpt_context = torch.tensor(x['gpt_context'], dtype=torch.int64).to(self.device)
        self.context_in = self.context
        self.decoder_output = torch.tensor(x['dec_out'], dtype=torch.int64).to(self.device)# in testing phase, only used to calc loss for printing.
        if training:
            #dec_in: summary; dec_out: summary with <PAD> as golden
            self.decoder_input = torch.tensor(x['dec_in'], dtype=torch.int64).to(self.device)
            self.decoder_len = x['dec_len']
            self.decoder_field_input = torch.tensor(x['dec_field'], dtype=torch.int64).to(self.device)
            self.decoder_pos_input = torch.tensor(x['dec_pos'], dtype=torch.int64).to(self.device)
            self.decoder_rpos_input = torch.tensor(x['dec_rpos'], dtype=torch.int64).to(self.device)
        else:
            pass

        '''
        context_outputs=define_decoder_arch(context_in),即GPT2生成下一个单词的logits,presents&hidden.
        context_in为context后接gpt_context，分别为：
        context：bill rigby , fullname : william rigby , birth date : 9 june 1921 , birth place : chester , england , death date : 01 june 2010 , currentclub : greasby , wirral , england , position : goalkeeper , youthyears : c 1937 -- 1939 , youthclubs : chester , years : c 1937 -- 1939 , clubs : chester , caps : 1 , goals : 0 , article title : bill rigby ( footballer ) ,
        gpt_context: Biography :
        context不足长度则前面补empty
        decoder_input: summary
        decoder_output: padded decoder_input右移位，用于计算loss。
        encoder_input: 表中内容对应token，内容：akoo nana solo_singer file  akoo nana.jpg william ato ankrah akoo nana 07 july 1979 ghana , accra ghanaian vocals dancehall , hiplife singer , songwriter , rapper 2012 -- present 4ever records ( 2009 -- present ) samini , stonebwoy , akoo nana
        '''
        #print(self.encoder_input[0], self.encoder_field[0], self.decoder_input[0], self.context[0])
        '''
        print("=================feed dict data shape==============")

        print("encoder_input shape: ", self.encoder_input.shape)
        print("encoder_field shape: " , self.encoder_field.shape)
        print("encoder_pos shape: " , self.encoder_pos.shape)
        print("encoder_rpos shape: " , self.encoder_rpos.shape)
        print("decoder_input shape: " , self.decoder_input.shape)
        print("decoder_output shape: " , self.decoder_output.shape)
        print("context shape: " , self.context.shape)
        print("gpt_context shape: " , self.gpt_context.shape)
        print("decoder_field_input shape: " ,  self.decoder_field_input.shape)
        print("decoder_pos_input shape: " ,  self.decoder_pos_input.shape)
        print("decoder_rpos_input shape: " ,  self.decoder_rpos_input.shape)

        print("===================================================")
        '''
    def forward(self, x, mode):#iterator输入x
        if mode=='train' or mode=='valid':
            self.create_feed_dict(x,training=True)

            enc_outputs, field_pos, enc_state = self.encoder(self.encoder_input, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)
            # gpt_context_in(context cat gpt_context), enc_output,field, encoder_input, summary, summary_pos
            # dec_out是vocab_logits.
            dec_out, cover_loss, copy_gate_loss= self.decoder(self.context_in.t(), field_pos, enc_outputs, self.encoder_input.t(),
                                    self.decoder_input.t(), self.decoder_len, self.decoder_field_input.t(), self.decoder_pos_input.t(), self.decoder_rpos_input.t())
            #print(self.decoder_output[0])
            #print("pred out: ", torch.argmax(dec_out,dim=-1)[0])
            #TODO: When change baseline, unquote this
            #dec_out, cover_loss, copy_gate_loss = self.decoder(self.context_in.t(), field_pos, enc_outputs,self.encoder_input.t(), enc_state, self.decoder_pos_input.t())

            #calculate loss
            loss = self.basic_loss(torch.log(dec_out.transpose(1,2)), self.decoder_output)#[batch_size, ground_len+1]
            cover_loss = cover_loss * self.coverage_penalty
            copy_gate_loss = self.copy_gate_penalty * copy_gate_loss#[batch_size, seq_len]

            #print(torch.sum(loss).item(), torch.sum(copy_gate_loss).item(), torch.sum(cover_loss).item())
            if self.use_coverage:
                loss = loss + cover_loss

            if self.use_copy_gate:
                loss = loss + copy_gate_loss
            #print("=====================Iter loss=====================")

            #3个loss都是[batch_size, 1]的tensor,直接backward即可
            return dec_out, torch.sum(loss), torch.sum(cover_loss), torch.sum(copy_gate_loss)

            #return dec_out, loss, cover_loss, copy_gate_loss
        else:
            self.create_feed_dict(x, training=False)

            enc_outputs, field_pos, enc_state = self.encoder(self.encoder_input, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)

            # gpt_context_in(context cat gpt_context), enc_output,field, encoder_input, summary, summary_pos
            # dec_out是vocab_logits.
            dec_out = self.decoder.greedy_decode(self.context_in.t(), field_pos, enc_outputs,
                                               self.encoder_input.t())
            return dec_out, torch.tensor(0), torch.tensor(0), torch.tensor(0)

class PromptModel(nn.Module):
    def __init__(self, config):
        super(PromptModel, self).__init__()
        self.device = config.device
        self.decoder = Decoder_prompt(config).to(self.device)
        #self.gen = Decoder_for_test(config).to(device)
        self.encoder = FGateEncoder(config, self.decoder.embedding, self.decoder.fembedding,self.decoder.pembedding).to(self.device)
        self.config = config
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.use_coverage = config.use_coverage
        self.use_copy_gate = config.use_copy_gate
        self.copy_gate_penalty = config.copy_gate_penalty
        self.coverage_penalty = config.coverage_penalty
        #MODIFIED
        #self.basic_loss = nn.CrossEntropyLoss(reduction='none')
        self.basic_loss = nn.NLLLoss(reduction='none')



    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference
            dec_in: gold truth
            dec_out: gold truth + eos*1
        Returns:
            feed_dict
        """

        self.encoder_input = torch.tensor(x['enc_in'], dtype=torch.int64).to(self.device)
        self.encoder_field = torch.tensor(x['enc_fd'], dtype=torch.int64).to(self.device)
        self.encoder_len = torch.tensor(x['enc_len'], dtype=torch.int64).to(self.device)
        self.encoder_pos = torch.tensor(x['enc_pos'], dtype=torch.int64).to(self.device)
        self.encoder_rpos = torch.tensor(x['enc_rpos'], dtype=torch.int64).to(self.device)
        self.context = torch.tensor(x['context_bart'], dtype=torch.int64).to(self.device)
        self.decoder_output = torch.tensor(x['dec_out'], dtype=torch.int64).to(self.device)# in testing phase, only used to calc loss for printing.
        if training:
            #dec_in: summary; dec_out: summary with <PAD> as golden
            self.decoder_input = torch.tensor(x['dec_in'], dtype=torch.int64).to(self.device)
            self.decoder_len = x['dec_len']
            self.decoder_field_input = torch.tensor(x['dec_field'], dtype=torch.int64).to(self.device)
            self.decoder_pos_input = torch.tensor(x['dec_pos'], dtype=torch.int64).to(self.device)
            self.decoder_rpos_input = torch.tensor(x['dec_rpos'], dtype=torch.int64).to(self.device)
        else:
            pass

        '''
        context_outputs=define_decoder_arch(context_in),即GPT2生成下一个单词的logits,presents&hidden.
        context_in为context后接gpt_context，分别为：
        context：bill rigby , fullname : william rigby , birth date : 9 june 1921 , birth place : chester , england , death date : 01 june 2010 , currentclub : greasby , wirral , england , position : goalkeeper , youthyears : c 1937 -- 1939 , youthclubs : chester , years : c 1937 -- 1939 , clubs : chester , caps : 1 , goals : 0 , article title : bill rigby ( footballer ) ,
        gpt_context: Biography :
        context不足长度则前面补empty
        decoder_input: summary
        decoder_output: padded decoder_input右移位，用于计算loss。
        encoder_input: 表中内容对应token，内容：akoo nana solo_singer file  akoo nana.jpg william ato ankrah akoo nana 07 july 1979 ghana , accra ghanaian vocals dancehall , hiplife singer , songwriter , rapper 2012 -- present 4ever records ( 2009 -- present ) samini , stonebwoy , akoo nana
        '''

        '''
        print("=================feed dict data shape==============")

        print("encoder_input shape: ", self.encoder_input.shape)
        print("encoder_field shape: " , self.encoder_field.shape)
        print("encoder_pos shape: " , self.encoder_pos.shape)
        print("encoder_rpos shape: " , self.encoder_rpos.shape)
        print("decoder_input shape: " , self.decoder_input.shape)
        print("decoder_output shape: " , self.decoder_output.shape)
        print("context shape: " , self.context.shape)
        print("gpt_context shape: " , self.gpt_context.shape)
        print("decoder_field_input shape: " ,  self.decoder_field_input.shape)
        print("decoder_pos_input shape: " ,  self.decoder_pos_input.shape)
        print("decoder_rpos_input shape: " ,  self.decoder_rpos_input.shape)

        print("===================================================")
        '''
    def forward(self, x, mode):#iterator输入x
        if mode=='train':
            self.create_feed_dict(x,training=True)

            enc_outputs, field_pos, enc_state = self.encoder(self.encoder_input, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)
            # gpt_context_in(context cat gpt_context), enc_output,field, encoder_input, summary, summary_pos
            # dec_out是vocab_logits.
            dec_out, cover_loss, copy_gate_loss= self.decoder(self.context.t(), field_pos, enc_outputs, self.encoder_input.t(),
                                    self.decoder_input.t(), self.decoder_len, self.decoder_field_input.t(), self.decoder_pos_input.t(), self.decoder_rpos_input.t())
            #print(self.decoder_output[0])
            #print("pred out: ", torch.argmax(dec_out,dim=-1)[0])
            #TODO: When change baseline, unquote this
            #dec_out, cover_loss, copy_gate_loss = self.decoder(self.context_in.t(), field_pos, enc_outputs,self.encoder_input.t(), enc_state, self.decoder_pos_input.t())

            #calculate loss
            loss = self.basic_loss(torch.log(dec_out.transpose(1,2)), self.decoder_output)#[batch_size, ground_len+1]
            cover_loss = cover_loss * self.coverage_penalty
            copy_gate_loss = self.copy_gate_penalty * copy_gate_loss#[batch_size, seq_len]

            #print(torch.sum(loss).item(), torch.sum(copy_gate_loss).item(), torch.sum(cover_loss).item())
            if self.use_coverage:
                loss = loss + cover_loss

            if self.use_copy_gate:
                loss = loss + copy_gate_loss
            #print("=====================Iter loss=====================")

            #3个loss都是[batch_size, 1]的tensor,直接backward即可
            return dec_out, torch.sum(loss), torch.sum(cover_loss), torch.sum(copy_gate_loss)

            #return dec_out, loss, cover_loss, copy_gate_loss
        else:
            self.create_feed_dict(x, training=False)

            enc_outputs, field_pos, enc_state = self.encoder(self.encoder_input, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)

            # gpt_context_in(context cat gpt_context), enc_output,field, encoder_input, summary, summary_pos
            # dec_out是vocab_logits.
            dec_out = self.decoder.greedy_decode(self.context.t(), field_pos, enc_outputs,
                                               self.encoder_input.t())
            return dec_out, torch.tensor(0), torch.tensor(0), torch.tensor(0)


class BartforT2T(nn.Module):
    def __init__(self, config):
        super(BartforT2T, self).__init__()
        self.device = config.device
        self.seq2seq = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(self.device)
        ptuning_config = PrefixTuningConfig(cross_prefix=False, leave_out=[12,13,14,15,16,17,18,19,20,21,22,23])
        adapter_config = MAMConfig(prefix_tuning=ptuning_config)
        
        self.seq2seq.add_adapter("mamconfig", config=adapter_config)
        #MAM: prefixtuning(bottleneck dim=30, i.e., prepend 30 tokens) + FFN旁边的SPA
        self.seq2seq.train_adapter("mamconfig")
        #print(self.seq2seq)
        self.tokenizer=encoder.get_encoder("BART")
        self.config = config
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.use_coverage = config.use_coverage
        self.use_copy_gate = config.use_copy_gate
        self.copy_gate_penalty = config.copy_gate_penalty
        self.coverage_penalty = config.coverage_penalty



    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference
            dec_in: gold truth
            dec_out: gold truth + eos*1
        Returns:
            feed_dict
        """

        self.encoder_input = torch.tensor(x['enc_in'], dtype=torch.int64).to(self.device)
        self.encoder_field = torch.tensor(x['enc_fd'], dtype=torch.int64).to(self.device)
        self.encoder_len = torch.tensor(x['enc_len'], dtype=torch.int64).to(self.device)
        self.encoder_pos = torch.tensor(x['enc_pos'], dtype=torch.int64).to(self.device)
        self.encoder_rpos = torch.tensor(x['enc_rpos'], dtype=torch.int64).to(self.device)
        self.context = torch.tensor(x['context_bart'], dtype=torch.int64).to(self.device)
        self.decoder_output = torch.tensor(x['dec_out'], dtype=torch.int64).to(self.device)# in testing phase, only used to calc loss for printing.
        if training:
            #dec_in: summary; dec_out: summary with <PAD> as golden
            self.decoder_input = torch.tensor(x['dec_in'], dtype=torch.int64).to(self.device)
            self.decoder_len = x['dec_len']
            self.decoder_field_input = torch.tensor(x['dec_field'], dtype=torch.int64).to(self.device)
            self.decoder_pos_input = torch.tensor(x['dec_pos'], dtype=torch.int64).to(self.device)
            self.decoder_rpos_input = torch.tensor(x['dec_rpos'], dtype=torch.int64).to(self.device)
        else:
            pass

        '''
        context_outputs=define_decoder_arch(context_in),即GPT2生成下一个单词的logits,presents&hidden.
        context_in为context后接gpt_context，分别为：
        context：bill rigby , fullname : william rigby , birth date : 9 june 1921 , birth place : chester , england , death date : 01 june 2010 , currentclub : greasby , wirral , england , position : goalkeeper , youthyears : c 1937 -- 1939 , youthclubs : chester , years : c 1937 -- 1939 , clubs : chester , caps : 1 , goals : 0 , article title : bill rigby ( footballer ) ,
        gpt_context: Biography :
        context不足长度则前面补empty
        decoder_input: summary
        decoder_output: padded decoder_input右移位，用于计算loss。
        encoder_input: 表中内容对应token，内容：akoo nana solo_singer file  akoo nana.jpg william ato ankrah akoo nana 07 july 1979 ghana , accra ghanaian vocals dancehall , hiplife singer , songwriter , rapper 2012 -- present 4ever records ( 2009 -- present ) samini , stonebwoy , akoo nana
        '''
        
        '''
        print("=================feed dict data shape==============")

        print("encoder_input shape: ", self.encoder_input.shape)
        print("encoder_field shape: " , self.encoder_field.shape)
        print("encoder_pos shape: " , self.encoder_pos.shape)
        print("encoder_rpos shape: " , self.encoder_rpos.shape)
        print("decoder_input shape: " , self.decoder_input.shape)
        print("decoder_output shape: " , self.decoder_output.shape)
        print("context shape: " , self.context.shape)
        print("gpt_context shape: " , self.gpt_context.shape)
        print("decoder_field_input shape: " ,  self.decoder_field_input.shape)
        print("decoder_pos_input shape: " ,  self.decoder_pos_input.shape)
        print("decoder_rpos_input shape: " ,  self.decoder_rpos_input.shape)

        print("===================================================")
        '''

    def forward(self, x, mode):#iterator输入x
        if mode=='train':# or mode=='valid':
            self.create_feed_dict(x,training=True)
            context_mask = torch.ones_like(self.context)
            context_mask = context_mask.masked_fill(self.context.eq(self.config.pad), 0.0).type(torch.FloatTensor).to(self.device)
            out=self.seq2seq(input_ids=self.context, attention_mask=context_mask, decoder_input_ids=self.decoder_input,
                    labels=self.decoder_output)
            dec_out = out.logits
            lm_loss = out.loss
            #print("pred out: ", torch.argmax(dec_out,dim=-1)[0])
            #TODO: When change baseline, unquote this
            #dec_out, cover_loss, copy_gate_loss = self.decoder(self.context_in.t(), field_pos, enc_outputs,self.encoder_input.t(), enc_state, self.decoder_pos_input.t())
            #3个loss都是[batch_size, 1]的tensor,直接backward即可
            return dec_out, torch.sum(lm_loss), torch.tensor(0), torch.tensor(0)

            #return dec_out, loss, cover_loss, copy_gate_loss
        else:
            self.create_feed_dict(x, training=False)
            context_mask = torch.ones_like(self.context)
            context_mask = context_mask.masked_fill(self.context.eq(self.config.pad), 0.0).type(torch.FloatTensor).to(self.device)
            dec_out=self.seq2seq.generate(input_ids=self.context, attention_mask=context_mask, num_beams=5, max_length=self.max_length)
            return dec_out[:,2:], torch.tensor(0), torch.tensor(0), torch.tensor(0)


class PromptSelector(nn.Module):
    def __init__(self, config):
        super(PromptSelector, self).__init__()
        self.device = config.device
        
        self.seq2seq = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(self.device)
        ptuning_config = PrefixTuningConfig(cross_prefix=False, leave_out=[12,13,14,15,16,17,18,19,20,21,22,23])
        adapter_config = MAMConfig(prefix_tuning=ptuning_config)
        
        self.seq2seq.add_adapter("mamconfig", config=adapter_config)
        self.seq2seq.train_adapter("mamconfig")
        
        self.selector = SequenceLabeling(config).to(self.device)
        self.config = config
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size



    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference
            dec_in: gold truth
            dec_out: gold truth + eos*1
        Returns:
            feed_dict
        """

        self.encoder_input = torch.tensor(x['enc_in'], dtype=torch.int64).to(self.device)
        self.encoder_field = torch.tensor(x['enc_fd'], dtype=torch.int64).to(self.device)
        self.encoder_len = torch.tensor(x['enc_len'], dtype=torch.int64).to(self.device)
        self.encoder_pos = torch.tensor(x['enc_pos'], dtype=torch.int64).to(self.device)
        self.encoder_rpos = torch.tensor(x['enc_rpos'], dtype=torch.int64).to(self.device)
        self.context = torch.tensor(x['context_bart'], dtype=torch.int64).to(self.device)
        self.decoder_output = torch.tensor(x['dec_out'], dtype=torch.int64).to(self.device)# in testing phase, only used to calc loss for printing.
        if training:
            #dec_in: summary; dec_out: summary with <PAD> as golden
            self.decoder_input = torch.tensor(x['dec_in'], dtype=torch.int64).to(self.device)
            self.decoder_len = x['dec_len']
            self.decoder_field_input = torch.tensor(x['dec_field'], dtype=torch.int64).to(self.device)
            self.decoder_pos_input = torch.tensor(x['dec_pos'], dtype=torch.int64).to(self.device)
            self.decoder_rpos_input = torch.tensor(x['dec_rpos'], dtype=torch.int64).to(self.device)
            self.key_content = torch.tensor(x['enc_keylabel'], dtype=torch.int64).to(self.device)
        else:
            pass

        '''
        context_outputs=define_decoder_arch(context_in),即GPT2生成下一个单词的logits,presents&hidden.
        context_in为context后接gpt_context，分别为：
        context：bill rigby , fullname : william rigby , birth date : 9 june 1921 , birth place : chester , england , death date : 01 june 2010 , currentclub : greasby , wirral , england , position : goalkeeper , youthyears : c 1937 -- 1939 , youthclubs : chester , years : c 1937 -- 1939 , clubs : chester , caps : 1 , goals : 0 , article title : bill rigby ( footballer ) ,
        gpt_context: Biography :
        context不足长度则前面补empty
        decoder_input: summary
        decoder_output: padded decoder_input右移位，用于计算loss。
        encoder_input: 表中内容对应token，内容：akoo nana solo_singer file  akoo nana.jpg william ato ankrah akoo nana 07 july 1979 ghana , accra ghanaian vocals dancehall , hiplife singer , songwriter , rapper 2012 -- present 4ever records ( 2009 -- present ) samini , stonebwoy , akoo nana
        '''

        '''
        print("=================feed dict data shape==============")

        print("encoder_input shape: ", self.encoder_input.shape)
        print("encoder_field shape: " , self.encoder_field.shape)
        print("encoder_pos shape: " , self.encoder_pos.shape)
        print("encoder_rpos shape: " , self.encoder_rpos.shape)
        print("decoder_input shape: " , self.decoder_input.shape)
        print("decoder_output shape: " , self.decoder_output.shape)
        print("context shape: " , self.context.shape)
        print("gpt_context shape: " , self.gpt_context.shape)
        print("decoder_field_input shape: " ,  self.decoder_field_input.shape)
        print("decoder_pos_input shape: " ,  self.decoder_pos_input.shape)
        print("decoder_rpos_input shape: " ,  self.decoder_rpos_input.shape)

        print("===================================================")
        '''
    def forward(self, x, mode):#iterator输入x
        if mode=='train' or mode=='valid':
            self.create_feed_dict(x,training=True)
            key_mask, sel_loss = self.selector(self.encoder_input, self.key_content, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)
            out=self.seq2seq(input_ids=self.encoder_input, attention_mask=key_mask, decoder_input_ids=self.decoder_input, labels=self.decoder_output)
            dec_out = out.logits
            lm_loss = out.loss
            loss=sel_loss+lm_loss
            return dec_out, torch.sum(loss), torch.sum(sel_loss), torch.tensor(0)

            #return dec_out, loss, cover_loss, copy_gate_loss
        else:
            self.create_feed_dict(x, training=False)

            enc_outputs, field_pos, enc_state = self.encoder(self.encoder_input, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)

            # gpt_context_in(context cat gpt_context), enc_output,field, encoder_input, summary, summary_pos
            # dec_out是vocab_logits.
            dec_out = self.decoder.greedy_decode(self.context.t(), field_pos, enc_outputs,
                                               self.encoder_input.t())
            return dec_out, torch.tensor(0), torch.tensor(0), torch.tensor(0)

class Selector(nn.Module):
    def __init__(self, config):
        super(Selector, self).__init__()
        self.device = config.device
        self.selector = SequenceLabeling(config).to(self.device)
        self.config = config
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size



    def create_feed_dict(self, x, training=True):
        """
        Create feed dict with placeholder keys for feeding x input to model
        Args:
            x: dict, input
            training: bool, for training or inference
            dec_in: gold truth
            dec_out: gold truth + eos*1
        Returns:
            feed_dict
        """

        self.encoder_input = torch.tensor(x['enc_in'], dtype=torch.int64).to(self.device)
        self.encoder_field = torch.tensor(x['enc_fd'], dtype=torch.int64).to(self.device)
        self.encoder_len = torch.tensor(x['enc_len'], dtype=torch.int64).to(self.device)
        self.encoder_pos = torch.tensor(x['enc_pos'], dtype=torch.int64).to(self.device)
        self.encoder_rpos = torch.tensor(x['enc_rpos'], dtype=torch.int64).to(self.device)
        self.context = torch.tensor(x['context_bart'], dtype=torch.int64).to(self.device)
        self.decoder_output = torch.tensor(x['dec_out'], dtype=torch.int64).to(self.device)# in testing phase, only used to calc loss for printing.
        if training:
            #dec_in: summary; dec_out: summary with <PAD> as golden
            self.decoder_input = torch.tensor(x['dec_in'], dtype=torch.int64).to(self.device)
            self.decoder_len = x['dec_len']
            self.decoder_field_input = torch.tensor(x['dec_field'], dtype=torch.int64).to(self.device)
            self.decoder_pos_input = torch.tensor(x['dec_pos'], dtype=torch.int64).to(self.device)
            self.decoder_rpos_input = torch.tensor(x['dec_rpos'], dtype=torch.int64).to(self.device)
            self.key_content = torch.tensor(x['enc_keylabel'], dtype=torch.int64).to(self.device)
        else:
            pass

        '''
        context_outputs=define_decoder_arch(context_in),即GPT2生成下一个单词的logits,presents&hidden.
        context_in为context后接gpt_context，分别为：
        context：bill rigby , fullname : william rigby , birth date : 9 june 1921 , birth place : chester , england , death date : 01 june 2010 , currentclub : greasby , wirral , england , position : goalkeeper , youthyears : c 1937 -- 1939 , youthclubs : chester , years : c 1937 -- 1939 , clubs : chester , caps : 1 , goals : 0 , article title : bill rigby ( footballer ) ,
        gpt_context: Biography :
        context不足长度则前面补empty
        decoder_input: summary
        decoder_output: padded decoder_input右移位，用于计算loss。
        encoder_input: 表中内容对应token，内容：akoo nana solo_singer file  akoo nana.jpg william ato ankrah akoo nana 07 july 1979 ghana , accra ghanaian vocals dancehall , hiplife singer , songwriter , rapper 2012 -- present 4ever records ( 2009 -- present ) samini , stonebwoy , akoo nana
        '''

        '''
        print("=================feed dict data shape==============")

        print("encoder_input shape: ", self.encoder_input.shape)
        print("encoder_field shape: " , self.encoder_field.shape)
        print("encoder_pos shape: " , self.encoder_pos.shape)
        print("encoder_rpos shape: " , self.encoder_rpos.shape)
        print("decoder_input shape: " , self.decoder_input.shape)
        print("decoder_output shape: " , self.decoder_output.shape)
        print("context shape: " , self.context.shape)
        print("gpt_context shape: " , self.gpt_context.shape)
        print("decoder_field_input shape: " ,  self.decoder_field_input.shape)
        print("decoder_pos_input shape: " ,  self.decoder_pos_input.shape)
        print("decoder_rpos_input shape: " ,  self.decoder_rpos_input.shape)

        print("===================================================")
        '''
    def forward(self, x, mode):#iterator输入x
        if mode=='train_seqlabel':
            self.create_feed_dict(x,training=True)
            key_mask, sel_loss = self.selector(self.encoder_input, self.key_content, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)
            acc = torch.sum(key_mask.eq(self.key_content)*~self.encoder_pos.eq(0))/torch.sum(~self.encoder_pos.eq(0)).item()
            return key_mask, torch.sum(sel_loss), acc

            #return dec_out, loss, cover_loss, copy_gate_loss
        else:
            self.create_feed_dict(x,training=True)
            key_mask, sel_loss = self.selector(self.encoder_input, self.key_content, self.encoder_field, self.encoder_pos,
                                                             self.encoder_rpos)
            return key_mask, torch.sum(sel_loss), torch.sum(key_mask.eq(self.key_content)*~self.encoder_pos.eq(0)),torch.sum(~self.encoder_pos.eq(0)).item()
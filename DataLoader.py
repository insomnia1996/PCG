#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import encoder
import os
import torch

PAD_TOKEN_ID = 52

class Preprocessor:
    def __init__(self, data_dir, limits, bos, eos, empty):
        """
        Main dataloader
        Args:
            data_dir: str, path to data directory
            limits:
            eos: str, eos character
            empty:
        """
        self.data_dir = data_dir

        self.limits = limits
        self.man_text_len = 150
        self.man_summary_len = 85
        self.bos = bos
        self.eos = eos
        self.empty = empty
        start_time = time.time()

        print('Reading datasets ...')
        self.train_set = self.load_data('train')
        self.test_set = self.load_data('test')
        self.dev_set = self.load_data('valid')
        print('Reading datasets consumes %.3f seconds' % (time.time() - start_time))

        # load fieldid2word list len 3
        self.fieldid2word = []
        with open(data_dir + "/field2word.txt") as f:
            for line in f:
                word_list = line.strip().split("\t")[1].split(" ")
                wordid_list = [int(tmp) for tmp in word_list]
                assert len(wordid_list) == 3
                self.fieldid2word.append(wordid_list)

        self.fieldid2word = torch.Tensor(self.fieldid2word)


    def load_file(self, file_path):
        """
        Load file, limit to self.limits lines, convert to list of lists
        Args:
            file_path: str, file path

        Returns:
            List of lists of tokens
        """
        data = open(file_path).read().strip().split('\n')
        if self.limits > 0:
            data = data[:self.limits]
        #print("data length: ", len(data))# 100
        #print("sample: ", data[0].strip().split(' '))
        
        tmp = [list(map(int, d.strip().split(' '))) if d else [] for d in data]
        return tmp
    def load_file2(self, file_path):
        """
        Load file, limit to self.limits lines, convert to list of lists
        Args:
            file_path: str, file path

        Returns:
            List of lists of tokens
        """
        data = open(file_path).read().strip().split('\n')
        if self.limits > 0:
            data = data[:self.limits]
        #print("data length: ", len(data))# 100
        #print("sample: ", data[0].strip().split(' '))
        tmp=[]
        for d in data:
            lst=[]
            for ele in d.strip().split('\t'):
                lst.append( list(map(int,ele.split(' '))) )
            tmp.append(lst)
        return tmp

    def load_data(self, split):
        """
        Load all data
        Args:
            split: str, one of 'train', 'test' or 'valid'

        Returns:
            Dict of data
        """
        subdir = os.path.join(self.data_dir, split)
        file_path_suffixes = {'summary': '.summary.id',
                              'text': '.box.val.id',
                              'field': '.box.lab.id',
                              'key': '.key.id',
                              'value': '.value.id',
                              'pos': '.box.pos',
                              'rpos': '.box.rpos',
                              'dec': '_summary_field_id.txt',
                              'dec_pos': '_summary_pos.txt',
                              'plan': '_content_plan.txt',#是每个key对应的key label id
                              'plan_id': '_content_plan_id.txt',
                              'dec_rpos': '_summary_rpos.txt',
                              'enc_keylabel': '.box.iskey',
                              'cont_path': '.context.id'}#与gpt_context相似，都是模板内容，例如song name: Lemon, author: XXX, artist: XXX. etc.

        all_data = {}
        for fp in file_path_suffixes.keys():
            file_path = os.path.join(subdir, split + file_path_suffixes[fp])
            if fp!='key' and fp!='value':
                all_data[fp] = self.load_file(file_path)
            else:
                all_data[fp] = self.load_file2(file_path)

        return all_data


class DataLoader:
    def __init__(self, data, domain, batch_size=64, shuffle=True, man_text_len=512,
                 man_summary_len=85, bos=0, eos=50256, empty=2, tkn="117M"):
        """
        Main dataloader
        Args:
            data_dir: dict, all the data
            batch_size: int, batch size
            shuffle: bool, Whether to shuffle data
            domain: str, domain name
        """
        self.data = data
        self.domain = domain
        self.batch_size = batch_size
        self.man_text_len = man_text_len
        self.man_summary_len = man_summary_len
        self.bos = bos
        self.eos = eos
        self.empty = empty
        self.data_size = len(data['summary'])
        self.enc = encoder.get_encoder(tkn)
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1
        if shuffle:
            self.shuffle_all_data()
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count < self.num_batches:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        """
        Shuffle all data
        Returns:
            None
        """
        data_size = len(self.data['summary'])
        shuffle_indices = np.random.permutation(np.arange(data_size))
        for fp in self.data.keys():
            self.data[fp] = np.array(self.data[fp], dtype=object)[shuffle_indices]
        return

    def get_zipped_batch(self, data, start_index, end_index):
        """
        Get zipped batch of data given start and end index
        Args:
            data: Dict of data
            start_index: int, start index
            end_index: int, end index

        Returns:
            Iterable of batch data
        """
        return zip(data['summary'][start_index:end_index],
                   data['text'][start_index:end_index],
                   data['field'][start_index:end_index],
                   data['key'][start_index:end_index],
                   data['value'][start_index:end_index],
                   data['pos'][start_index:end_index],
                   data['rpos'][start_index:end_index],
                   data['dec'][start_index:end_index],
                   data['dec_pos'][start_index:end_index],
                   data['dec_rpos'][start_index:end_index],
                   data['plan'][start_index:end_index],
                   data['plan_id'][start_index:end_index],
                   data['enc_keylabel'][start_index:end_index],
                   data['cont_path'][start_index:end_index])

    def get_batch(self):
        #在这里修改batch输入的结构
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)

        max_summary_len = max([len(sample) for sample in self.data['summary'][start_index:end_index]])
        max_text_len = max([len(sample) for sample in self.data['text'][start_index:end_index]])
        max_cont_len = max([len(sample) for sample in self.data['cont_path'][start_index:end_index]])
        max_order_len = max([len(sample) for sample in self.data['plan_id'][start_index:end_index]])
        max_plan_len = max([len(sample) for sample in self.data['plan'][start_index:end_index]])
        

        batch_data = {'enc_in': [],'key': [], 'value': [], 'enc_fd': [], 'enc_pos': [], 'enc_rpos': [], 'enc_len': [],
                      'dec_in': [], 'dec_len': [], 'dec_out': [], 'oov_map': [], 'dec_field': [],
                      'dec_pos': [], 'dec_rpos': [], 'plan':[], 'plan_id':[],
                      'gpt_context': [], 'context': [], 'context_bart': [], 'enc_keylabel': []}

        data_subset = self.get_zipped_batch(self.data, start_index, end_index)
        gpt_context = " Summarize this table | "
        gpt_context = self.enc.encode(gpt_context)

        for summary, text, field, key, value, pos, rpos, dec_field, dec_pos, dec_rpos, plan, plan_id, key_label, cont_text in data_subset:
            summary_len = len(summary)
            text_len = len(text)
            cont_len = len(cont_text)
            order_len = len(plan_id)
            plan_len = len(plan)
            pos_len = len(pos)
            rpos_len = len(rpos)
            assert text_len == len(field)
            assert text_len == len(key_label)
            assert pos_len == len(field)
            assert rpos_len == pos_len
            assert len(key)==len(value)
            assert len(dec_field) == len(summary)

            gold = summary + [self.eos] + [self.empty] * (max_summary_len - summary_len + 1)
            '''
            [21953, 35066,   318,   257,  5337,   416,  6072,  2731,   374,    13,
            1619,  1092,   837,  3199,   287,  4343,   416,  1097,  2487,  7933,
            69,   837,   281, 29122,   286, 37441,   261, 12407,  1448,   764,
             2,     1]
            '''
            # context = [self.eos] * (max_summary_len - summary_len) + summary
            summary = [self.bos] + summary + [self.eos] + [self.empty] * (max_summary_len - summary_len)
            '''
            [0, 21953, 35066,   318,   257,  5337,   416,  6072,  2731,   374,
            13,  1619,  1092,   837,  3199,   287,  4343,   416,  1097,  2487,
            7933,    69,   837,   281, 29122,   286, 37441,   261, 12407,  1448,
            764,     2]
            '''
            # key value
            #for i in range(len(key)):



            # empty field id is 0
            dec_field = [0] + dec_field + [0] * (max_summary_len - summary_len + 1)
            dec_pos = [0] + dec_pos + [0] * (max_summary_len - summary_len + 1)
            dec_rpos = [0] + dec_rpos + [0] * (max_summary_len - summary_len + 1)
            plan_id = plan_id + [PAD_TOKEN_ID] * (max_order_len - order_len)

            context = [self.empty] * (max_cont_len - cont_len) + gpt_context + cont_text #length: max_cont_len+len(gpt_context)
            context_bart = plan + cont_text + [self.empty] * (max_cont_len + max_plan_len - plan_len - cont_len) #length: max_cont_len+max_plan_len
            text = text + [self.empty] * (max_text_len - text_len)
            key_label = key_label + [0] * (max_text_len - text_len)
            field = field + [0] * (max_text_len - text_len)
            pos = pos + [0] * (max_text_len - text_len)
            rpos = rpos + [0] * (max_text_len - text_len)

            if max_text_len > self.man_text_len:
                text = text[:self.man_text_len]
                field = field[:self.man_text_len]
                pos = pos[:self.man_text_len]
                rpos = rpos[:self.man_text_len]
                text_len = min(text_len, self.man_text_len)

            elif max_cont_len > self.man_text_len:
                context = context[-self.man_text_len-len(gpt_context):]
                context_bart = context_bart[:self.man_text_len+max_plan_len]


            # OOM
            if max_summary_len + 2 > self.man_summary_len:
                gold = gold[:self.man_summary_len]
                summary = summary[:self.man_summary_len]

                # context = context[-self.man_summary_len:]

                dec_field = dec_field[:self.man_summary_len]
                dec_pos = dec_pos[:self.man_summary_len]
                dec_rpos = dec_rpos[:self.man_summary_len]
                summary_len = min(summary_len + 2, self.man_summary_len)
            else:
                summary_len = summary_len + 2

            batch_data['enc_in'].append(text)  # value
            #enc_in为表中内容对应token，内容：akoo nana solo_singer file  akoo nana.jpg william ato ankrah akoo nana 07 july 1979 ghana , accra ghanaian vocals dancehall , hiplife singer , songwriter , rapper 2012 -- present 4ever records ( 2009 -- present ) samini , stonebwoy , akoo nana
            batch_data['enc_keylabel'].append(key_label)  # value
            batch_data['enc_len'].append(text_len)  # value length
            batch_data['enc_fd'].append(field)  # field
            batch_data['enc_pos'].append(pos)  # field p+
            batch_data['enc_rpos'].append(rpos)  # field p-
            batch_data['dec_in'].append(summary)  # summary
            batch_data['dec_len'].append(summary_len)  # summary len
            batch_data['dec_out'].append(gold)  # padded summary
            batch_data['dec_field'].append(dec_field)  # masked summary
            batch_data['dec_pos'].append(dec_pos)  # summary pos
            batch_data['dec_rpos'].append(dec_rpos)  # summary rpos
            batch_data['plan'].append(plan)
            batch_data['plan_id'].append(plan_id)
            batch_data['key'].append(key)
            batch_data['value'].append(value)
            batch_data['gpt_context'].append(gpt_context)  # box for gpt input with domain name
            batch_data['context'].append(context)  # padded context
            batch_data['context_bart'].append(context_bart)  # padded context
            
            '''
            print("context:==============")
            for context in batch_data['context_bart']:
                print(context)
            '''
        return batch_data

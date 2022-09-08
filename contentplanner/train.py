# coding=utf-8
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import encoder
from config import BartConfig
from DataLoader import Preprocessor, DataLoader

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def parse_config():
    parser = argparse.ArgumentParser()
    #planner configuration
    parser.add_argument("--root_path", default="../data_release/", help="full path of data folder")
    parser.add_argument("--domain",default='books',help='domain name')
    parser.add_argument("--tkn_name",default='BART',help='tokenizer name')
    parser.add_argument("--n_sample", type=int, default=100, help="Number of training samples.")
    parser.add_argument("--gpu", default=0, help='gpu number', type=int)
    # model configuration
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--crf_low_rank", type=int)
    parser.add_argument("--crf_beam_size", type=int)
    # mini-batch training configuration
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    # pre-training configuration
    parser.add_argument("--total_steps", type=int, 
        help="total effective training steps")
    parser.add_argument("--print_every", type=int, 
        help="how many update steps to print one intermediate result")
    parser.add_argument("--save_every", type=int, 
        help="how many update steps to save one model")
    # learning configuration
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--mle_loss_weight", type=float) # 0.5
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        print ('Using single GPU training.')
    else:
        pass
    args = parse_config()
    # create data paths
    root_path = args.root_path
    gold_path_valid = os.path.join(root_path, args.domain, 'original_data', 'valid.summary')
    gold_path_test = os.path.join(root_path, args.domain, 'original_data', 'test.summary')
    field_vocab_file = os.path.join(root_path, "human_books_songs_films_field_vocab.txt")
    processed_data_dir = os.path.join(root_path, args.domain, "processed_data_{}".format(args.n_sample))
    table_path_valid = os.path.join(processed_data_dir, "valid", 'valid.box.parent')
    table_path_test = os.path.join(processed_data_dir, "test", 'test.box.parent')
    device = torch.device('cuda')
    model_name = args.model_name
    
    print ('Loading data...')
    config = BartConfig(deviceid=args.gpu)
    enc = encoder.get_encoder(args.tkn_name)
    bos=config.bos
    eos = config.eos
    empty = config.pad

    print ('Initializaing model...')
    #from utlis import load_special_tokens
    #special_token_list = load_special_tokens(args.special_token_path, args.min_slot_key_cnt)
    from contentplanner import ContentPlanner
    model = ContentPlanner(model_name, enc, args.crf_low_rank, args.crf_beam_size)#special_token_list)
    print ('Model initialized!')

    preprocessed_data = Preprocessor(processed_data_dir, 0, bos, eos, empty)
    train_data = DataLoader(preprocessed_data.train_set, args.domain, batch_size=args.batch_size_per_gpu, 
                      shuffle=True, bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)
    dev_data = DataLoader(preprocessed_data.dev_set, args.domain, batch_size=args.batch_size_per_gpu, 
                      shuffle=True, bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)
    print ('Data loaded.')

    from trainer import model_training
    print ('############################################################')
    print ('Start Training...')
    if cuda_available:
        model = model.to(device)
    else:
        pass
    print ('Model loaded') 
    total_steps, print_every, save_every = args.total_steps, args.print_every, args.save_every
    ckpt_save_path = args.save_path_prefix
    model = model_training(args, train_data, dev_data, model, total_steps, print_every, save_every, 
        ckpt_save_path, cuda_available, device)
    print ('Training stage completed!')
    print ('############################################################')

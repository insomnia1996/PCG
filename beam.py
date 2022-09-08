import torch
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

def create_masks(src, trg, opt):
    
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
        # MODIFIED
        #trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask
    
def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
        k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if opt.device >= 0:
        np_mask = np_mask.cuda()
    return np_mask

def get_mask(words, pad_index, punct = False, punct_list = None):
    '''
    get the mask of a sentence, mask all <pad>
    the starting of sentence is <ROOT>, mask[0] is always False
    :param words: sentence
    :param pad_index: pad index
    :param punct: whether to ignore the punctuation, when punct is False, take all the punctuation index to False(for evaluation)
    punct is True for getting loss
    :param punct_list: only used when punct is False
    :return:
    For example, for a sentence:  <ROOT>     no      ,       it      was     n't     Black   Monday  .
    when punct is True,
    The returning value is       [False    True     True    True    True    True    True    True    True]
    when punct is False,
    The returning value is      [False    True     False    True    True    True    True    True    False]
    '''
    mask = words.ne(pad_index)
    mask[:, 0] = False
    if not punct:
        puncts = words.new_tensor(punct_list)
        mask &= words.unsqueeze(-1).ne(puncts).all(-1)
    return mask

def init_beam(src, e_output, field_pos, model, opt):
    bos_token=0
    trg_input = torch.LongTensor([[bos_token]])
    if opt.device >= 0:
        trg_input = trg_input.to(opt.device)
    
    
    out = model.decoder(e_output, trg_input)
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_strlen).long()#(topk, seq_len)
    if opt.device >= 0:
        outputs = outputs.cuda()
    outputs[:, 0] = bos_token
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.device >= 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    
    return outputs, e_outputs, log_scores
    
def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    if opt.device >= 0:
        outputs = outputs.cuda()
    
    trg_mask = nopeak_mask(1, opt)
    
    out = model.out(model.decoder(outputs,
    e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long()
    if opt.device >= 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.device >= 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    #topk x max_seq_len, topk x vocab_size 
    
    probs, ix = out.data.topk(k)
    
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1) # 概率累加
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    

    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)

        out = model.out(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])

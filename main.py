from config import GPT2Config,BartConfig
from DataLoader import DataLoader, Preprocessor
import torch
import json, time, random, os ,re, math
import numpy as np
from tqdm import tqdm, trange
import encoder
from model import Model, BartforT2T, PromptModel, PromptSelector, Selector
from table_text_eval.table_text_eval import parent
from utils import get_current_git_version, bleu_score, write_log, parent_score
import argparse


torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()

parser.add_argument("--root_path", default="./data_release/", help="full path of data folder")
parser.add_argument("--domain",default='books',help='domain name')
parser.add_argument("--tkn_name",default='117M',help='tokenizer name')
parser.add_argument("--output_path", default="./output/", help="full path of saved output")
parser.add_argument("--mode",default='train',help='train or test')
parser.add_argument("--gpu", default=0, help='gpu number', type=int)
# training
parser.add_argument("--batch_size", default=1, type=int, help="Batch size of train set.")
parser.add_argument("--batch_update", type=int, default=20, help="apply gradients after steps")
parser.add_argument("--epoch", type=int, default=50, help="Number of training epoch.")
parser.add_argument("--n_sample", type=int, default=100, help="Number of training samples.")
parser.add_argument("--learning_rate", type=float, default=1e-5,help='learning rate')
parser.add_argument("--clip_value", default=10.0,help='gradient clipping value')
# logging
parser.add_argument("--report", type=int, default=10, help='report valid results after some steps')
parser.add_argument("--report_loss", type=int, default=10, help='report loss results after some steps')
args = parser.parse_args()

#Set the random seeds for reproducability.
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if args.tkn_name=='117M' or args.tkn_name=='gpt2':
    config = GPT2Config(deviceid=args.gpu)
elif args.tkn_name=='BART':
    config = BartConfig(deviceid=args.gpu)
input_dim= config.vocab_size
field_dim= config.field_vocab
position_dim=config.position_vocab
hidden_size=config.hidden_size
field_size=config.field_size
pos_size=config.pos_size

#argparse config
batch_size=args.batch_size
mode = args.mode
# create output paths
if mode == "train":

    results_path = os.path.join(args.output_path, args.domain, "results")
    saved_model_path = os.path.join(args.output_path, args.domain, "saved_model")
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(saved_model_path, exist_ok=True)
else:
    saved_model_path = os.path.join(args.output_path, args.domain, "saved_model")
    results_path = os.path.join(args.output_path, args.domain, "inference_results")
    os.makedirs(results_path, exist_ok=True)

log_file = os.path.join(args.output_path, 'log.txt')


# create data paths
root_path = args.root_path
gold_path_valid = os.path.join(root_path, args.domain, 'original_data', 'valid.summary')
gold_path_test = os.path.join(root_path, args.domain, 'original_data', 'test.summary')
field_vocab_file = os.path.join(root_path, "human_books_songs_films_field_vocab.txt")
processed_data_dir = os.path.join(root_path, args.domain, "processed_data_{}".format(args.n_sample))
table_path_valid = os.path.join(processed_data_dir, "valid", 'valid.box.parent')
table_path_test = os.path.join(processed_data_dir, "test", 'test.box.parent')

# bpe vocab
last_best = 0.0
enc = encoder.get_encoder(args.tkn_name)
bos=config.bos
eos = config.eos
empty = config.pad

#TODO:改一下这个内容
def acc_sent(logits, labels, ignore_index):#(bsz, seq_len, vocab)&(bsz, seq_len)
    bsz=logits.size(0)# default is 1
    assert logits.size(1)==labels.size(1)
    if len(logits.size())==3:
        _, logits = logits.max(dim=-1)
    cnt=torch.sum(labels!=ignore_index)
    corr=0

    for i in range(bsz):
        for index in labels[i]:
            if index in logits[i] and index!=ignore_index:
                corr+=1
    return corr/cnt

def train(preprocessed_data,model):

    train_iterator = DataLoader(preprocessed_data.train_set, args.domain,
                                batch_size=args.batch_size, shuffle=True, bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)

    k = 0
    record_k = 0
    record_loss_k = 0
    total_loss, start_time = 0.0, time.time()
    record_loss = 0.0
    record_copy_loss = 0.0
    record_cov_loss = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.7,weight_decay=0.02)
    model.train()
    best_bleu = 0.0

    for tst in range(args.epoch):#, desc="Epoch"):
        #if tst<16:
        config.use_coverage = False
        #else:
        #    config.use_coverage = True
        train_iterator.reset()
        optimizer.zero_grad()
        print("========================Epoch %d=====================" %tst)
        with tqdm(train_iterator, desc="Iteration") as pbar:
            for x in pbar:
                
                dec_out, total_loss, cover_loss, copy_gate_loss = model(x, 'train')
                acc = acc_sent(dec_out, model.decoder_output, config.pad)
                pbar.set_postfix({'loss' : '{0:1.3f}'.format(total_loss.item()), 'acc' : '{0:1.2f}'.format(acc)})
                if len(dec_out.size())==3:
                    dec_out = torch.argmax(dec_out, dim=-1, keepdim=False)#[batch_size, ground_len+1]
                for ind, summary in enumerate(np.array(dec_out.detach().cpu())):
                    #tokenize标准不一致
                    summary = list(summary)#[ground_len+1]
                    if eos in summary:
                        summary = summary[:summary.index(eos)] if summary[0] != eos else [eos]
                    cont  = enc.decode(summary)#,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    cont = cont.replace("\n", " ")
                    cont2 = model.decoder_output[ind].tolist()

                    if eos in cont2:
                        cont2 = cont2[:cont2.index(eos)] if cont2[0] != eos else [eos]
                    cont2 = enc.decode(cont2)
                    cont2 = cont2.replace("\n", " ")

                    print("ori sentence: %s  out sentence: %s  acc.: %f." % (cont2, cont, acc))
                total_loss.backward()
                #print("batch loss: %.2f." %total_loss.item())
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

                # TODO: debugging grad exploding and vanishing
                #for name, params in model.named_parameters():
                #    if params.grad is not None:
                #       params.register_hook(lambda grad: torch.clamp(grad, -args.clip_value, args.clip_value))
                #       print("-->name:", name, "-->max_grad:", params.grad.abs().max(), "-->min_grad:", params.grad.abs().min(), "-->max_value:", params.abs().max(), "-->min_value:", params.abs().min())
                optimizer.step()
                k += 1

                #TODO also add to tensorboard
                if k % args.batch_update == 0:

                    record_loss += total_loss
                    record_copy_loss += copy_gate_loss
                    record_cov_loss += cover_loss
                    record_k += 1
                    record_loss_k += 1

                    if record_loss_k > 1 and record_loss_k % args.report_loss == 0:
                        write_log(log_file, "%d : loss = %.3f, copyloss = %.3f, covloss = %.3f" % \
                            (record_k, record_loss / record_loss_k, record_copy_loss / record_loss_k,
                             record_cov_loss / record_loss_k))
                        print("%d : loss = %.3f, copyloss = %.3f, covloss = %.3f" % \
                            (record_k, record_loss / record_loss_k, record_copy_loss / record_loss_k,
                             record_cov_loss / record_loss_k))
                        record_loss = 0.0
                        record_copy_loss = 0.0
                        record_cov_loss = 0.0
                        record_loss_k = 0

                    if record_k > 1 and record_k % args.report == 0:
                        write_log(log_file,"Round: %d."%(record_k //args.report))
                        print("Round: %d."%(record_k //args.report))
                        cost_time = time.time() - start_time
                        write_log(log_file, "%d : time = %.3f " % (record_k // args.report, cost_time))
                        print("%d : time = %.3f " % (record_k // args.report, cost_time))
                        start_time = time.time()
                        torch.save(model.state_dict(),saved_model_path+'/model_tmp.pt')
                        global mode 
                        mode = 'valid'
                        validation_result, bleu_score, parent_core = evaluate(preprocessed_data, model)
                        write_log(log_file, validation_result)
                        print(validation_result)
                        if bleu_score > best_bleu:
                            # save model
                            torch.save(model.state_dict(),saved_model_path+'/model.pt')
                            best_bleu = bleu_score
                            print("Temporary best bleu score is :%4f" %best_bleu)
                        mode = 'train'

def train_seqlabel(preprocessed_data, model):

    train_iterator = DataLoader(preprocessed_data.train_set, args.domain,
                                batch_size=args.batch_size, shuffle=True, bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)

    k = 0
    record_k = 0
    record_loss_k = 0
    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.7,weight_decay=0.02)
    model.train()
    best_acc = 0.0

    for tst in range(args.epoch):#, desc="Epoch"):
        train_iterator.reset()
        optimizer.zero_grad()
        print("========================Epoch %d=====================" %tst)
        with tqdm(train_iterator, desc="Iteration") as pbar:
            for x in pbar:
        #for x in train_iterator:
                
                key_mask, total_loss, acc = model(x, 'train_seqlabel')
                pbar.set_postfix({'loss' : '{0:1.3f}'.format(total_loss.item()), 'acc' : '{0:1.2f}'.format(acc)})
                total_loss.backward()
                optimizer.step()
                k += 1

                #TODO also add to tensorboard
                if k % args.batch_update == 0:
                    record_k += 1
                    record_loss_k += 1

                    if record_k > 1 and record_k % args.report == 0:
                        write_log(log_file,"Round: %d."%(record_k //args.report))
                        print("Round: %d."%(record_k //args.report))
                        cost_time = time.time() - start_time
                        write_log(log_file, "%d : time = %.3f " % (record_k // args.report, cost_time))
                        print("%d : time = %.3f " % (record_k // args.report, cost_time))
                        start_time = time.time()
                        torch.save(model.state_dict(),saved_model_path+'/select_model_tmp.pt')
                        global mode 
                        data_iterator = DataLoader(preprocessed_data.dev_set,
                                    args.domain, batch_size=args.batch_size, shuffle=False,
                                    bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)
                        acc_cnt,acc_ttl=0,0
                        for x in data_iterator:
                            key_mask, total_loss, cnt, ttl = model(x, 'valid')
                            acc_cnt+=cnt
                            acc_ttl+=ttl
                        print("test accuracy: ", acc_cnt/acc_ttl)
                        if acc > best_acc:
                            # save model
                            torch.save(model.state_dict(),saved_model_path+'/select_model.pt')
                            best_acc = acc

def evaluate(preprocessed_data,model):
    if mode == 'valid' or mode == 'train':
        gold_path = gold_path_valid
        table_path = table_path_valid
        data_iterator = DataLoader(preprocessed_data.dev_set,
                                    args.domain, batch_size=args.batch_size, shuffle=False,
                                    bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)
    else:
        gold_path = gold_path_test
        table_path = table_path_test
        data_iterator = DataLoader(preprocessed_data.test_set,
                                   args.domain, batch_size=args.batch_size, shuffle=False,
                                   bos=bos, eos=eos, empty=empty, tkn=args.tkn_name)

    pred_list = []
    pred_unk = []

    out_bpe = open(os.path.join(results_path, mode + "_summary_bpe.txt"), "w")
    out_real = open(os.path.join(results_path,  mode + "_summary_clean.txt"), "w")

    write_log(log_file,"=========================Evaluating===============================")
    # save model
    if mode == 'valid' or mode == 'train':
        saved_model_path_cnt = saved_model_path + '/model_tmp.pt'
    else:
        saved_model_path_cnt = saved_model_path + '/model.pt'

    checkpoint = torch.load(saved_model_path_cnt)
    model.load_state_dict(checkpoint)
    #torch.no_grad()
    model.eval()
    for x in data_iterator:

            dec_out, total_loss, cover_loss, copy_gate_loss = model(x, mode)
            if len(dec_out.size())==3:
                dec_out = torch.argmax(dec_out, dim=-1, keepdim=False)#[batch_size, ground_len+1]
            for ind, summary in enumerate(np.array(dec_out.cpu())):
                summary = list(summary)#[ground_len+1]
                if eos in summary:
                    summary = summary[:summary.index(eos)] if summary[0] != eos else [eos]
                real_sum = enc.decode(summary)

                bpe_sum = " ".join([enc.decoder[tmp] if tmp in enc.decoder else enc.added_tokens_decoder[tmp] for tmp in summary])
                real_sum = real_sum.replace("\n", " ")
                cont2 = model.decoder_output[ind].tolist()
                if eos in cont2:
                    cont2 = cont2[:cont2.index(eos)] if cont2[0] != eos else [eos]
                cont2 = enc.decode(cont2)
                cont2 = cont2.replace("\n", " ")
                print("ori sentence: %s  pred sentence: %s" % (cont2, real_sum))
                pred_list.append(real_sum)
                pred_unk.append(bpe_sum)

                out_real.write(real_sum + '\n')
                out_bpe.write(bpe_sum + '\n')

    out_bpe.close()
    out_real.close()

    # new bleu
    bleu_copy = bleu_score(gold_path, os.path.join(results_path,  mode + "_summary_clean.txt"))
    copy_result = "with copy BLEU: %.4f\n" % bleu_copy
    parent_f = parent_score(gold_path, os.path.join(results_path,  mode + "_summary_clean.txt"), table_path)
    parent_result = "with PARENT-F: %.4f\n" % parent_f

    result = copy_result+"\n"+parent_result

    return result, bleu_copy, parent_f



def main():
    #git_commit_id = get_current_git_version()
    #write_log(log_file, "GIT COMMIT ID: " + git_commit_id)
    # set limit to 0, fetch all data

    preprocessed_data = Preprocessor(processed_data_dir, 0, config.bos, config.eos, config.pad)                      
    device = config.device
    if args.tkn_name=='117M' or args.tkn_name=='gpt2':
        model = Model(config).to(device)
    elif args.tkn_name=='BART':
        #model = PromptModel(config).to(device)
        model = BartforT2T(config).to(device)
        #model = PromptSelector(config).to(device)
    if mode == "train":
        write_log(log_file,"train process")
        train(preprocessed_data,model)
    else:
        write_log(log_file,"evaluate process")
        validation_result, bleu_score, parent_score = evaluate(preprocessed_data, model)
        write_log(log_file, validation_result)


if __name__ == '__main__':
    main()

'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''


# context: 表格模板化输出：title : once an eagle, author : anton myrer, country : united states, language : english, genre : war, publisher : holt, rinehart, and winston, publication date : 1968, media type : print ( hardback paperback ), pages : 1312, isbn :.,
# enc_in: 表格value输出：once an eagle anton myrer united states english war holt, rinehart, and winston 1968 print ( hardback paperback ) 1312.
# field_in: 表格field输出：name name name empty empty publication_date genre empty empty country country empty author author author author empty
# dec_in: 表格内容总结：once an eagle is a 1968 war novel by american author anton myrer.

def load_weight(model, state_dict):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="")

    # Make sure we are still sharing the output and input embeddings after loading weights
    model.set_tied()
    return model

import time, os, sys, shutil, io, subprocess, re
import numpy as np
import zipfile
import logging

from table_text_eval.table_text_eval import parent

logger = logging.getLogger(__name__)

def parent_score(labels_file, predictions_path, table_file):
    try:
        cmd=os.path.join(os.path.dirname(os.path.realpath(__file__)),"table_text_eval","table_text_eval.py")
        parent_out = subprocess.check_output(['python', cmd, "--references", labels_file, 
                                               "--generations", predictions_path, 
                                               "--tables", table_file],
                                               stderr=subprocess.STDOUT)
        parent_out = parent_out.decode("utf-8")
        parent_score = re.search(r"F-score = (.+)", parent_out).group(1)
        return float(parent_score)*100
    except subprocess.CalledProcessError as error:
        if error.output is not None:
            msg = error.output.strip()
            logger.warning(
                "parent file returned non-zero exit code: {}".format(msg))
        return None

def bleu_score(labels_file, predictions_path):
    bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [bleu_script, labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)

    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        logger.warning(
            "{} script returned non-zero exit code: {}".format(bleu_script, msg))
      return None

def read_word2vec_zip(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    zfile = zipfile.ZipFile(word2vec_file)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        for line in ifile:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension

def read_word2vec(word2vec_file):
    wordvec_map = {}
    num_words = 0
    dimension = 0
    with open(word2vec_file, "r") as f:
        for line in f:
            line = line.strip()
            #print line
            entries = line.split(' ')
            if len(entries) == 2:
                continue
            word = entries[0].strip()
            vec = map(float, entries[1:])

            if word in wordvec_map:
                print ("Invalid word in embedding. Does not matter.")
                continue
            # assert word not in wordvec_map
            assert dimension == 0 or dimension == len(vec)

            wordvec_map[word] = np.array(vec)
            num_words += 1
            dimension = len(vec)

    return wordvec_map, num_words, dimension

def load_vocab(vocab_file):
    vocab = {}

    vocab['<_PAD>'] = 0
    vocab['<_START_TOKEN>'] = 1
    vocab['<_END_TOKEN>'] = 2
    vocab['<_UNK_TOKEN>'] = 3

    cnt = 4
    with open(vocab_file, "r") as v:
        for line in v:
            if len(line.strip().split()) > 1:
                word = line.strip().split()[0]
                ori_id = int(line.strip().split()[1])
                if word not in vocab:
                    vocab[word] = (cnt + ori_id)

    return vocab

def create_init_embedding(vocab_file, extend_vocab_size, word2vec_file, emblen):
    '''
    create initial embedding for text relation words.
    words not in word2vec file initialized to random.

    key_map['PAD'] = 0
    key_map['START_TOKEN'] = 1
    key_map['END_TOKEN'] = 2
    key_map['UNK_TOKEN'] = 3
    '''
    from gensim.models import KeyedVectors
    vocab = load_vocab(vocab_file)
    print("vocab len: ", len(vocab))

    init_embedding = np.random.uniform(-np.sqrt(3), np.sqrt(3), size = (len(vocab) + extend_vocab_size, emblen))

    if word2vec_file.endswith('.gz'):
        word2vec_map = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    elif word2vec_file.endswith('.zip'):
        word2vec_map, num_words, dimension = read_word2vec_zip(word2vec_file)
    else:
        word2vec_map, num_words, dimension = read_word2vec(word2vec_file)

    num_covered = 0

    for word in vocab:
        if word in word2vec_map:
            vec = word2vec_map[word]
            if len(vec) != emblen:
                raise ValueError("word2vec dimension doesn't match.")
            init_embedding[vocab[word], :] = vec
            num_covered += 1

    unk_vec = init_embedding[3, :]
    for ind in range(len(vocab), len(init_embedding)):
        init_embedding[ind, :] = unk_vec

    ## embedding for pad
    # init_embedding[0][:] = np.zeros(emblen)

    print ("word2vec covered: %d" % num_covered)
    return init_embedding

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def write_word(pred_list, save_dir, name):
    ss = open(save_dir + name, "w+")
    for item in pred_list:
        ss.write(" ".join(item) + '\n')


def get_current_git_version():
    import git
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')

if __name__=='__main__':
    f1=parent_score('./data_release/humans/original_data/test.summary', 
    './output/humans/inference_results/test_summary_clean.txt',
    './data_release/humans/processed_data_50/test/test.box.parent')
    f2=bleu_score('./data_release/humans/original_data/test.summary', 
    './output/humans/inference_results/test_summary_clean.txt')
    print(f1,f2)
"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, model_name, errors='replace'):
        if "117m" in model_name.lower():
            self.eos_token_id=50256
            self.bos_token_id=50256
        elif "bart" in model_name.lower():
            self.eos_token_id=2
            self.bos_token_id=0
            self.pad_token_id=1
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""\s*<unused[0-9]+>|\s*<table2text>|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            ### handle oov
            for bpe_token in self.ignore_unk(self.bpe(token)).split(' '):
                
                if bpe_token in self.encoder:
                    bpe_tokens.append(self.encoder[bpe_token])
                else:
                    bpe_tokens.append(self.encoder["empty"]) # OOV
        return bpe_tokens
    
    def tokenize(self, text):
        bpe_tokens = []
        bpe_token_original = []
        for token in re.findall(self.pat, text):
            
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_token_original.extend(bpe_token for bpe_token in self.ignore_unk(self.bpe(token)).split(' '))
        return bpe_token_original
    
    def convert_tokens_to_ids(self, tokenlst):
        bpe_tokens=[]
        for bpe_token in tokenlst:
            if bpe_token in self.encoder:
                bpe_tokens.append(self.encoder[bpe_token])
            else:
                bpe_tokens.append(self.encoder["empty"]) # OOV
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
    
    def ignore_unk(self, text):
        pat = re.compile(r'Ġ?< un used [0-9]+ >|Ġ?< unused [0-9]+ >|Ġ?< table 2 text >')
        flag = re.search(pat, text)
        if flag:
            proc_text = ''.join(text.split()).strip(r'Ġ')
        else:
            proc_text = text
        return proc_text

    def __len__(self):
        return len(self.encoder)


def get_encoder(model_name):
    curdir = os.path.dirname(__file__)
    if "bart" in model_name.lower():
        with open(os.path.join(curdir, 'models', model_name, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(curdir, 'models', model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
        model_name=model_name,
    )
    elif "117m" in model_name.lower():
        with open(os.path.join(curdir, 'models', model_name, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(curdir, 'models', model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
        model_name=model_name,
    )

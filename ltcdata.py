import os
import codecs
import numpy as np
import pickle

import tensorflow.compat.v1 as tf

import BERT_tokenization as tokenizer

FNAME_VOCAB = "vocab.txt"
FNAME_VOCABOBJ = "uvocab_basic_corpus_unite.dat"
FNAME_TRAIN = "1.tsv"
FNAME_DEV = "2.tsv"
FNAME_EVAL = "0.tsv"

MASK = "[MASK]"
IDX_MASK = 3 # [wc, [EOS], [UNK], [MASK], <NIL>, ...]

NUM_LTC = 27
def get_sid_list(is_united):
    return [0] if is_united else [i for i in range(NUM_LTC)]

###
### Model output path
###
def make_str_of_setting(
    is_united=False, is_global=False, use_BERT_tokenizer=False):
    ret = "united" if is_united else "separated"
    if is_united:
        ret += "_global" if is_global else "_local"
    if use_BERT_tokenizer:
        ret += "_BERTtkn"
    
    return ret

def make_name_outdir(
    is_united=False, is_global=False, use_BERT_tokenizer=False, sid=0):
    ret = make_str_of_setting(is_united, is_global, use_BERT_tokenizer)
    ret = os.path.join("runs_ltc", ret, str(sid))

    return ret

def make_name_indir(is_united=False):
    return os.path.join("data_ltc", "united" if is_united else "separated")

###
### sentence process
###
def parse_line(line):
    line = line.strip()

    tokens = line.split("\t")

    tid = int(tokens[0])
    sent = tokens[1]
    sid = int(tokens[2])

    return tid, sent, sid

###
### vocab process
###
def load_vocab(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        vocab_obj = pickle.load(f)
    vocab, ivocab = vocab_obj["vocab"], vocab_obj["ivocab"]

    # replace meta tokens
    vocab["[UNK]"] = vocab["<UNK>"]
    vocab["[EOS]"] = vocab["<EOS>"]
    ivocab[vocab["[UNK]"]] = "[UNK]"
    ivocab[vocab["[EOS]"]] = "[EOS]"
    del vocab["<UNK>"]
    del vocab["<EOS>"]

    # add mask
    vocab[MASK] = IDX_MASK
    ivocab[IDX_MASK] = MASK

    return vocab, ivocab

def tokenize(line, vocab):
    return [
        (vocab[w] if w in vocab else vocab["[UNK]"]) for w in line.split(" ")]

###
### example separator
###
def separate_united_examples(path_src="data_ltc/united/0.tsv", fps=[]):
    max_sid = 0
    max_tid_this = 0
    tid_offset = 0
    for line in codecs.open(path_src, "r", "utf-8"):
        tid, sent, sid = parse_line(line)        

        if sid != max_sid:
            tid_offset = max_tid_this+1
            max_sid = sid
        
        fps[sid].write("%d\t%s\t%d\n" % (tid-tid_offset, sent, sid))
        max_tid_this = max(tid, max_tid_this)

def separate_united_dataset(
    path_src="data_ltc/united", path_dst="data_ltc/separated", n_lts=27):
    
    for i in range(n_lts):
        try:
            os.makedirs(os.path.join(path_dst, str(i)))
        except FileExistsError:
            pass
    
    for i in range(3):
        fps = [codecs.open(os.path.join(
            path_dst, str(j), "%d.tsv" % i), "w", "utf-8") for j in range(n_lts)]

        separate_united_examples(os.path.join(path_src, "0", "%d.tsv" % i), fps)

        list(map(lambda x: x.close(), fps))

###
### example process
###
def load_examples(
    basedir, name, tkn, vocab=None, is_united=True, is_global=False, sid=0, max_len=128):
    """Make tuple of (examples, labels), each of which is numpy array."""
    if is_united:
        path = os.path.join(basedir, str(0), name)
    else:
        path = os.path.join(basedir, str(sid), name)

    examples = []
    labels = []
    max_sid = 0
    max_tid_this = 0
    tid_offset = 0
    for line in tf.io.gfile.GFile(path, "r"):
        tid, sent, sid = parse_line(line)

        if sid != max_sid:
            tid_offset = max_tid_this
            max_sid = sid

        if is_united is True and is_global is False:
            tid = tid - tid_offset

        if tkn:
            wids = tkn.convert_tokens_to_ids(tkn.tokenize(sent))
            wids = pad_ids(wids, 0, max_len)
        else:
            wids = tokenize(sent, vocab)
            wids = pad_ids(wids, vocab["[EOS]"], max_len)
        
        labels.append(tid)
        examples.append(wids)

    n_labels = len(np.unique(labels))
    
    return np.array(examples), np.eye(n_labels)[labels]

def pad_ids(wids, id_unk, max_len):
    return (wids + [id_unk for _ in range(max_len)])[:max_len]

###
### data loader
###
def load_data(
    data_dir="data_ltc/separated",
    is_united=True, is_global=False, sid=0,
    with_train=False, with_dev=False, with_eval=False, 
    use_BERT_tokenizer=False, max_len=128):
    """Load vocab and examples."""
    if use_BERT_tokenizer:
        tkn = tokenizer.FullTokenizer(
            os.path.join(data_dir, FNAME_VOCAB),
            do_tokenize_chinese_chars=False, do_lower_case=False,
            metawords=["[MASK]", "[UNK]"]
        )
        ret = {
            "tokenizer": tkn,
            "vocab": None,
            "num_vocab": len(tkn.vocab)
        }
    else:
        vocab, ivocab = load_vocab(os.path.join(data_dir, FNAME_VOCABOBJ))
        ret = {
            "tokenizer": None,
            "vocab": vocab,
            "ivocab": ivocab,
            "num_vocab": len(ivocab)-1
        }

    if with_train:
        ret["train"] = load_examples(
            data_dir, FNAME_TRAIN, ret["tokenizer"], ret["vocab"],
            is_united, is_global, sid, max_len)

    if with_dev:
        ret["dev"] = load_examples(
            data_dir, FNAME_DEV, ret["tokenizer"], ret["vocab"],
            is_united, is_global, sid, max_len)
    
    if with_eval:
        ret["eval"] = load_examples(
            data_dir, FNAME_EVAL, ret["tokenizer"], ret["vocab"],
            is_united, is_global, sid, max_len)
    
    return ret

if __name__ == "__main__":
    separate_united_dataset()
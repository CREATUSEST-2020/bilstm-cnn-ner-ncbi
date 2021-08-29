import os
import re
import random
import logging
from collections import Counter

import gdown
import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

logger = logging.getLogger(__name__)


def get_test_texts(args):
    sentences = []
    sentence = []
    with open(os.path.join(args.data_dir, args.test_file), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if not line and len(sentence) > 1:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
            else:
                word = line.split()
                sentence.append(word)
        if len(sentence) > 1:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
        s_list = []
        for s in sentences:
            word_list = []
            label_list = []
            for ite in s:
                word_list.append(ite[0])
                label_list.append(ite[1])
            final = (" ").join(word_list)
            s_list.append(final)
    return s_list


def build_vocab(args):
    """
    Build vocab from train set
    Write all the tokens in vocab. When loading the vocab, limit the size of vocab at that time
    """
    # Read all the files
    total_words, total_chars = [], []

    with open(os.path.join(args.data_dir, args.train_file), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line)>0:
                words, _ = line.split()
                words = words.split()
                for word in words:
                    for char in word:
                        total_chars.append(char)
                total_words.extend(words)

    if not os.path.exists(args.vocab_dir):
        os.mkdir(args.vocab_dir)

    word_vocab, char_vocab = [], []

    word_vocab_path = os.path.join(args.vocab_dir, "word_vocab")
    char_vocab_path = os.path.join(args.vocab_dir, "char_vocab")

    word_counts = Counter(total_words)
    word_vocab.append("PAD")
    word_vocab.append("UNK")
    word_vocab.extend([x[0] for x in word_counts.most_common()])
    logger.info("Total word vocabulary size: {}".format(len(word_vocab)))

    with open(word_vocab_path, 'w', encoding='utf-8') as f:
        for word in word_vocab:
            f.write(word + "\n")

    char_counts = Counter(total_chars)
    char_vocab.append("PAD")
    char_vocab.append("UNK")
    char_vocab.extend([x[0] for x in char_counts.most_common()])
    logger.info("Total char vocabulary size: {}".format(len(char_vocab)))

    with open(char_vocab_path, 'w', encoding='utf-8') as f:
        for char in char_vocab:
            f.write(char + "\n")

    # Set the exact vocab size
    # If the original vocab size is smaller than args.vocab_size, then set args.vocab_size to original one
    with open(word_vocab_path, 'r', encoding='utf-8') as f:
        word_lines = f.readlines()
        args.word_vocab_size = min(len(word_lines), args.word_vocab_size)

    with open(char_vocab_path, 'r', encoding='utf-8') as f:
        char_lines = f.readlines()
        args.char_vocab_size = min(len(char_lines), args.char_vocab_size)

    logger.info("args.word_vocab_size: {}".format(args.word_vocab_size))
    logger.info("args.char_vocab_size: {}".format(args.char_vocab_size))

def cap_features(s):
    """
    Capitalization feature:
    1 = low caps
    2 = all caps
    3 = first letter caps
    4 = one capital (not first letter)
    """
    if s.lower() == s:
        return 1
    elif s.upper() == s:
        return 2
    elif s[0].upper() == s[0]:
        return 3
    else:
        return 4
    
def load_vocab(args):
    word_vocab_path = os.path.join(args.vocab_dir, "word_vocab")
    char_vocab_path = os.path.join(args.vocab_dir, "char_vocab")

    if not os.path.exists(word_vocab_path):
        logger.warning("Please build word vocab first!")
        return

    if not os.path.exists(char_vocab_path):
        logger.warning("Please build char vocab first!")
        return

    word_vocab = dict()
    char_vocab = dict()
    word_ids_to_tokens = []
    char_ids_to_tokens = []

    # Load word vocab
    with open(word_vocab_path, "r", encoding="utf-8") as f:
        # Set the exact vocab size
        # If the original vocab size is smaller than args.vocab_size, then set args.vocab_size to original one
        word_lines = f.readlines()
        args.word_vocab_size = min(len(word_lines), args.word_vocab_size)

        for idx, line in enumerate(word_lines[:args.word_vocab_size]):
            line = line.strip()
            word_vocab[line] = idx
            word_ids_to_tokens.append(line)

    # Load char vocab
    with open(char_vocab_path, "r", encoding="utf-8") as f:
        char_lines = f.readlines()
        args.char_vocab_size = min(len(char_lines), args.char_vocab_size)
        for idx, line in enumerate(char_lines[:args.char_vocab_size]):
            line = line.strip()
            char_vocab[line] = idx
            char_ids_to_tokens.append(line)

    return word_vocab, char_vocab, word_ids_to_tokens, char_ids_to_tokens



def get_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def load_label_vocab(args):
    label_vocab = dict()
    for idx, label in enumerate(get_labels(args)):
        label_vocab[label] = idx

    return label_vocab


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)


def adjust_learning_rate(optimizer, lr, global_step):
    """
    shrink learning rate for pytorch
    """
    min_lrate = 0.000005
    if lr >= min_lrate and global_step >=5:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


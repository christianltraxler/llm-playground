import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset

import config
from llm import LanguageModel


torch.manual_seed(1234)


def main():
    # initialize language model
    language_model = LanguageModel()

    # load tiny shakespeare dataset
    data = load_dataset("tiny_shakespeare")

    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    data = torch.tensor(encode(text), dtype=torch.long)

    # train the model on the dataset
    language_model.train(data)



if __name__ == '__main__':
    main()

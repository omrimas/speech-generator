import os
import numpy as np
import bcolz
import pickle
import torch
import torch.nn as nn

glove_path = os.path.join("data", "glove")


# glove_type = '6B'
# glove_dim = 50


class GloveEmbedding(object):

    def __init__(self, voc):
        # # load and prepare glove mapping
        vectors = bcolz.open(f'{glove_path}/6B.50.dat')[:]
        words = pickle.load(open(f'{glove_path}/6B.50_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'{glove_path}/6B.50_idx.pkl', 'rb'))
        glove = {w: vectors[word2idx[w]] for w in words}

        # create weights_matrix for embedding layer
        self.weights_matrix = np.zeros((voc.num_words, 50))
        words_found = 0
        for i, word in voc.index2word.items():
            try:
                self.weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                self.weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))

    def createLayer(self, non_trainable=False):
        num_embeddings, embedding_dim = self.weights_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': torch.tensor(self.weights_matrix)})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer

import os
import time
import sys
import torch
from torch.autograd import Variable
from datetime import datetime
import data
import glove

started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
directory = "GENERATED/" + started_datestring + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

CORPUS_NAME = "Clinton-Trump Corpus"
USE_CUDA = torch.cuda.is_available()
MODEL_CHECKPOINT = "models/2019-03-26T12-20-25/model-LSTM-emsize-50-nhid_128-nlayers_6-batch_size_20-epoch_25.pt"
WORDS_TO_GEN = 100
TEMPRATURE = 1
SWITCH_WORDS = False

while (True):
    with open(MODEL_CHECKPOINT, 'rb') as f:
        model = torch.load(f)
        if USE_CUDA:
            model.cuda()
        else:
            model.cpu()

        corpus = data.Corpus(CORPUS_NAME)
        glove_embedding = glove.GloveEmbedding(corpus.vocabulary)
        ntokens = corpus.vocabulary.num_words
        hidden = model.init_hidden(1)
        input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
        if USE_CUDA:
            input.data = input.data.cuda()

    words = ''

    now = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    tfn = directory + "/" + now + "_" + MODEL_CHECKPOINT.split("model-")[1] + ".txt"

    with open(tfn, 'w') as outf:

        for i in range(WORDS_TO_GEN):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(TEMPRATURE).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word = corpus.vocabulary.index2word[word_idx.item()]

            # set next input
            if SWITCH_WORDS:
                # print("getting different but similar word to '%s'" % (word))
                similar_words_idx = glove_embedding.getSimilarWordIdx(word_idx)
                word_idx = similar_words_idx[5]
                # for sim_word_idx in similar_words_idx:
                #     print(corpus.vocabulary.index2word[sim_word_idx])

            input.data.fill_(word_idx)

            if word == 'EOS':
                word = '\n'

            words += word + " "

            # outf.write(word + ('\n' if i % 20 == 19 else ' '))

            # if i % args.log_interval == 0:
            # print('Generated {}/{} words'.format(i+1, args.words), end='\r')

        words = "\n\n" + words.split('\n', 1)[0] + "\n\n" + "\n".join(words.splitlines()[1:])

        # SCREEN OUTPUT
        for char in words:
            time.sleep(0.001)
            sys.stdout.write(char)

        outf.write(words)
        outf.close()
        # print("\n\nsaved to: "+ tfn)

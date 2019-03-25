import os
import time
import sys
import torch
from torch.autograd import Variable
from datetime import datetime
import data

started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
directory = "GENERATED/" + started_datestring + '/'
if not os.path.exists(directory):
    os.makedirs(directory)

CORPUS_NAME = "Clinton-Trump Corpus"
USE_CUDA = torch.cuda.is_available()
MODEL_CHECKPOINT = "models/2019-03-25T15-36-29/model-LSTM-emsize-50-nhid_128-nlayers_6-batch_size_20-epoch_10.pt"
WORDS_TO_GEN = 100
TEMPRATURE = 1

while (True):
    with open(MODEL_CHECKPOINT, 'rb') as f:
        model = torch.load(f)
        if USE_CUDA:
            model.cuda()
        else:
            model.cpu()

        corpus = data.Corpus(CORPUS_NAME)
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
            input.data.fill_(word_idx)
            word = corpus.vocabulary.index2word[word_idx.item()]

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

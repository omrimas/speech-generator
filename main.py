import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unicodedata

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
speaker = "Clinton"
corpus_name = "Clinton-Trump Corpus"
corpus = os.path.join("data", corpus_name, speaker)
target_vocabulary = []


################# Util functions ################################
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"\<.*?\>", " ", s)
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

################################################################

class LSTMGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to vocabulary space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        voc_space = self.hidden2word(lstm_out.view(len(sentence), -1))
        word_scores = F.log_softmax(voc_space, dim=1)
        return word_scores

# Vocabulary class
class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.num_words = 0

    def addFile(self, filePath):
        with open(filePath, 'r') as f:
            for line in f:
                normalized_line = normalizeString(line)
                for word in normalized_line.split():
                    self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

def text2Seq(filePath):
    with open(os.path.join(corpus, filePath), 'r') as f:
        seqs = [normalizeString(line) for line in f]
        seqs = list(filter(lambda x: len(x) > 0, seqs))
    return ' '.join(seqs).split(' ')

def prepareSequence(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# prepare data for training
voc = Voc(corpus_name)
voc.addFile(os.path.join(corpus, "Clinton_2016-07-28.txt"))
# training_data = text2Seq("Clinton_2016-07-28.txt")
files = ["Clinton_2016-07-28.txt"]


model = LSTMGenerator(50, 64, voc.num_words)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    training_data = text2Seq(files[0])
    inputs = prepareSequence(training_data, voc.word2index)
    word_scores = model(inputs)
    print(word_scores)

for epoch in range(100):
    for file in files:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        training_data = text2Seq(file)
        speech_in = prepareSequence(training_data, voc.word2index)
        targets = torch.cat((speech_in[1:], speech_in[0:1]))


        # Step 3. Run our forward pass.
        word_scores = model(speech_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(word_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    training_data = ["like"]
    inputs = prepareSequence(training_data, voc.word2index)
    for i in range(10):
        word_scores = model(inputs)
        _, indices = torch.max(word_scores, 1)
        word_index = indices[0].item()
        print(voc.index2word[word_index])
        inputs = torch.LongTensor([word_index])


# dataX = []
# dataY = []

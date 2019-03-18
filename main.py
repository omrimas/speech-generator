import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unicodedata

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
speakers = ["Clinton"]
corpus_name = "Clinton-Trump Corpus"
# corpus = os.path.join("data", corpus_name, speaker)
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

# Default word tokens
PAD_token = 0  # Used for padding short sentences
EOS_token = 1  # End-of-sentence token


# Vocabulary class
class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", EOS_token: "EOS"}
        self.num_words = 2

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


class LSTMGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2)

        # The linear layer that maps from hidden state space to vocabulary space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, word, hidden=None):
        embeds = self.word_embeddings(word)

        rnn_output, hidden = self.lstm(embeds.view(1, 1, -1), hidden)
        voc_space = self.hidden2word(rnn_output)
        word_scores = F.softmax(voc_space, dim=2)
        return word_scores, hidden


def getSpeech(filePath):
    with open(filePath, 'r') as f:
        seqs = [normalizeString(line) for line in f]
        seqs = list(filter(lambda x: len(x) > 0, seqs))
    return ' '.join(seqs).split(' ')


def speech2indices(seq, word_to_ix):
    return [word_to_ix[w] for w in seq]


def getAllFiles(corpus_name, speakers):
    all_speech_files = []
    for speaker in speakers:
        speaker_path = os.path.join("data", corpus_name, speaker)
        speaker_files = [os.path.join(speaker_path, speaker_file) for speaker_file in os.listdir(speaker_path)]
        all_speech_files += speaker_files
    return all_speech_files


all_speech_files = getAllFiles(corpus_name, speakers)

# prepare data for training
voc = Voc(corpus_name)
for speech_file in all_speech_files:
    voc.addFile(speech_file)

model = LSTMGenerator(128, 128, voc.num_words)
model.to(device)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1):

    for file in all_speech_files[:10]:
        print(file)
        hidden = None
        current_speech = getSpeech(file)
        speech_indices = speech2indices(current_speech, voc.word2index)
        speech_indices.append(EOS_token)
        for i in range(len(speech_indices) - 1):
            if i % 3 == 0:
                hidden = None
            current_word = torch.tensor(speech_indices[i], device=device)
            model.zero_grad()
            word_scores, hidden = model(current_word, hidden)
            target = torch.tensor([speech_indices[i + 1]], device=device)
            loss = loss_function(word_scores.view(1, -1), target)
            loss.backward(retain_graph=True)
            optimizer.step()

with torch.no_grad():
    hidden = None
    seed = "the"
    generated_speech = [seed]
    input_word = torch.tensor([voc.word2index[seed]], device=device)

    for i in range(100):
        word_scores, hidden = model(input_word, hidden)
        indices = torch.argmax(word_scores, 2)
        word_index = indices[0].item()
        generated_speech.append(voc.index2word[word_index])
        input_word = torch.tensor([word_index], device=device)

    print(" ".join(generated_speech))

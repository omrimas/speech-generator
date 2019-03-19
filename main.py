import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unicodedata
import numpy as np

USE_CUDA = torch.cuda.is_available()
SEQUENCE_LEN = 2
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

    def forward(self, seq, hidden=None):
        embeds = self.word_embeddings(seq)

        rnn_output, hidden = self.lstm(embeds.view(len(seq), 1, -1), hidden)
        voc_space = self.hidden2word(rnn_output.view(len(seq), -1))
        word_scores = F.log_softmax(voc_space, dim=1)
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

for epoch in range(5):

    for file in all_speech_files[:2]:
        print(file)
        current_speech = getSpeech(file)
        speech_indices = speech2indices(current_speech, voc.word2index)

        for i in range(len(speech_indices) - SEQUENCE_LEN):
            current_seq = torch.tensor(speech_indices[i:i + SEQUENCE_LEN], dtype=torch.long, device=device)
            target = torch.tensor([speech_indices[i + SEQUENCE_LEN]], dtype=torch.long, device=device)
            model.zero_grad()
            word_scores, _ = model(current_seq)
            last_word_scores = word_scores[SEQUENCE_LEN - 1].view(1, -1)
            loss = loss_function(last_word_scores, target)
            loss.backward()
            optimizer.step()

with torch.no_grad():
    seed = ["the", "people"]
    generated_speech = seed
    input_seq = torch.tensor(speech2indices(seed, voc.word2index), dtype=torch.long, device=device)

    for i in range(100):
        word_scores, _ = model(input_seq)
        last_word_scores = word_scores[SEQUENCE_LEN - 1].view(1, -1)
        # chosen_index = np.random.choice(range(voc.num_words), voc.num_words, p=word_scores[SEQUENCE_LEN - 1].cpu())
        indices = torch.argmax(last_word_scores, 1)
        word_index = indices[0].item()
        generated_speech.append(voc.index2word[word_index])
        input_seq = torch.tensor(speech2indices(generated_speech[-2:], voc.word2index), dtype=torch.long, device=device)

    print(" ".join(generated_speech))

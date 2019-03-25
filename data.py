import re
import os
import unicodedata
import torch

speakers = ["Clinton", "Trump", "adams", "arthur", "bharrison", "buchanan", "bush", "carter", "cleveland", "coolidge",
            "eisenhower", "fdroosevelt", "fillmore", "ford", "garfield", "grant", "gwbush", "harding", "harrison",
            "hayes", "hoover", "jackson", "jefferson", "johnson", "jqadams", "kennedy", "lbjohnson", "lincoln",
            "madison", "mckinley", "monroe", "nixon", "obama", "pierce", "polk", "reagan", "roosevelt", "taft",
            "taylor", "truman", "tyler", "vanburen", "washington", "wilson"]
EOS_token = 0  # End-of-sentence token


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


def getAllFiles(corpus_name, speakers):
    all_speech_files = []
    for speaker in speakers:
        speaker_path = os.path.join("data", corpus_name, speaker)
        speaker_files = [os.path.join(speaker_path, speaker_file) for speaker_file in os.listdir(speaker_path)]
        all_speech_files += speaker_files
    return all_speech_files


################################################################

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {EOS_token: "EOS"}
        self.num_words = 1

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1


class Corpus(object):
    def __init__(self, corpus_name):
        self.vocabulary = Voc(corpus_name)

        self.train = torch.LongTensor([])
        all_files = getAllFiles(corpus_name, speakers)
        for file in all_files:
            self.train = torch.cat((self.train, self.tokenize(file)), 0)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the vocabulary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = normalizeString(line).split() + ['EOS']
                if len(words) > 1:
                    tokens += len(words)
                    for word in words:
                        self.vocabulary.addWord(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = normalizeString(line).split() + ['EOS']
                if len(words) > 1:
                    for word in words:
                        ids[token] = self.vocabulary.word2index[word]
                        token += 1

        return ids

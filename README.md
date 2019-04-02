# speech-generator
This example trains a multi-layer RNN (LSTM) on a language modeling task.  <br/>The trained model can then be used by the generate scripts to generate new text.


## Training
To model was trained on a corpus of 1080 american political speeches taken from here:  
http://www.thegrammarlab.com/?nor-portfolio=corpus-of-presidential-speeches-cops-and-a-clintontrump-corpus


###Word Embedding
For the word embedding layer, I used a pre-trained GLoVE vectors. 
More specifically, the "Wikipedia 2014 + Gigaword 5" version which is the smallest file (glove.6B.zip) was trained on a 
corpus of 6 billion tokens and contains a vocabulary of 400 thousand tokens.  
Also, I picked the 50-dimensional vectors which is the smallest one (there are also 100/200/300-dimensional vectors). 

The pre-trained vectors can be download from here:  
https://nlp.stanford.edu/projects/glove/

<sub>I also used the word embedding for generation, but not just as part of the model.</sub>

 
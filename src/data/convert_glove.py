from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Here is the code used to save the model in compressed format.
# For more technical details see the following links:
# https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.init_sims.html
# https://groups.google.com/g/gensim/c/OvWlxJOAsCo
# https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time/43067907

MODEL_NAME = input('Please type your model name here. The temp file model will be\
saved to /src/data/interim/<MODEL_NAME>_tmp.txt\n->')
compress = input('Would you like to compress the model? For compression references \
see this source code of this file, /src/data/compress_glove.py.\n(y/n) ->')

glove_file = '../../data/external/glove.840B.300d/glove.840B.300d.txt'
_ = glove2word2vec(glove_file, f'../../data/interim/{MODEL_NAME}_tmp.txt')

if compress == 'y':
    we_model = KeyedVectors.load_word2vec_format('../data/interim/glove_840_tmp.txt')
    we_model.init_sims(replace=True)
    we_model.save(f'../../data/interim/{MODEL_NAME}_norm')
    print(f'Model successfully saved to ../../data/interim/{MODEL_NAME}_norm')
    # The compressed model can be accessed by using:
    #we_model = KeyedVectors.load('../../data/interim/glove_840B_norm', mmap='r')

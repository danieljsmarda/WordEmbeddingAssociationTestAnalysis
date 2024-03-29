import numpy as np
import pdb
from sklearn.preprocessing import MinMaxScaler
from gensim.models import KeyedVectors, Word2Vec
# You need the next import to save a compressed glove model
#from gensim.scripts.glove2word2vec import glove2word2vec

from bias_calculation import get_matrices_from_term_lists
from utils import save_arrays, open_pickle, save_pickle, save_scalers

# Here is the code used to save the model in compressed format.
# For more technical details see the following links:
# https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.init_sims.html
# https://groups.google.com/g/gensim/c/OvWlxJOAsCo
# https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time/43067907

# Directly loading the we_model from the .txt file takes 10-15 minutes on a laptop.
# Loading the `norm` file takes less than 1 minute.
'''
glove_file = '../../data/external/glove.840B.300d/glove.840B.300d.txt'
_ = glove2word2vec(glove_file, '../data/interim/glove_840_tmp.txt')
we_model = KeyedVectors.load_word2vec_format('../data/interim/glove_840B_tmp.txt')
we_model.init_sims(replace=True)
we_model.save('../data/interim/glove_840B_norm')
'''

MODEL_NAME = input('Please type your model name, then press ENTER. This name will be \
appended to all filenames. For example, the experiment definitions will \
be loaded from ../../data/interim/<model_name>_experiment_definitions.pickle.\n\
->')

# Load the model quickly.
we_model = KeyedVectors.load(f'../../data/interim/{MODEL_NAME}_norm', mmap='r')
we_model = KeyedVectors.load(f'../../data/interim/{MODEL_NAME}_norm', mmap='r')

print('loading done!')
print(f'Total words in model: {len(we_model.vocab)}')

EXPERIMENT_DEFINITION_PATH = f'../../data/interim/{MODEL_NAME}_experiment_definitions.pickle'
RESULTS_FILEPATH = f'../../data/interim/{MODEL_NAME}_association_metric_exps.pickle'
SCALERS_FILEPATH = f'../../data/processed/{MODEL_NAME}_scalers.pickle'
THRESHOLD_BIASES_PATH_2NDORDER = f'../../data/processed/{MODEL_NAME}_threshold_biases_2ndorder.pickle'

def calculate_cosines_for_target_word_unscaled(word_vec, A_mtx, B_mtx):
    A_dot_v = np.dot(A_mtx, word_vec)
    B_dot_v = np.dot(B_mtx, word_vec)
    A_norms = np.multiply(np.linalg.norm(A_mtx, axis=1), np.linalg.norm(word_vec))
    B_norms = np.multiply(np.linalg.norm(B_mtx, axis=1), np.linalg.norm(word_vec))
    A_cosines = np.divide(A_dot_v, A_norms)
    B_cosines = np.divide(B_dot_v, B_norms)
    return np.mean(A_cosines), np.mean(B_cosines)

def calculate_cosines_for_all_words_unscaled(we_model, A_mtx, B_mtx):
    '''Computes the association metric, s(w,A,B).
    A_mtx, B_mtx: 2-D word vector arrays'''
    # We also tried an alternative implementation using the following lines
    # A_cosines_apply = np.apply_along_axis(lambda row: 1-cosine_distance(row, word_vec), 1, A_mtx)
    # B_cosines_apply = np.apply_along_axis(lambda row: 1-cosine_distance(row, word_vec), 1, B_mtx)
    # but we found that the norm-based implementation was faster.
    A_mtx_norm = A_mtx/np.linalg.norm(A_mtx, axis=1).reshape(-1,1)
    B_mtx_norm = B_mtx/np.linalg.norm(B_mtx, axis=1).reshape(-1,1)
    all_mtx_norm = we_model.vectors/np.linalg.norm(we_model.vectors, axis=1).reshape(-1,1)
    
    all_associations_to_A = np.dot(A_mtx_norm, all_mtx_norm.T)
    all_associations_to_B = np.dot(B_mtx_norm, all_mtx_norm.T)
    
    return np.mean(all_associations_to_A, axis=0), np.mean(all_associations_to_B, axis=0)

def add_quantile_ranges_to_dict(dct, biases):
    dct['QR_95'] = [np.percentile(biases, 2.5), np.percentile(biases, 97.5)]
    dct['QR_99'] = [np.percentile(biases, 0.5), np.percentile(biases, 99.5)]
    dct['QR_99.9'] = [np.percentile(biases, 0.05), np.percentile(biases, 99.95)]
    return dct


def get_2ndorder_association_metric_list_for_target_list(target_mtx, A_associations, B_associations, A_mtx, B_mtx, we_model, exp_num):
    
    #pdb.set_trace()
    all_associations = np.concatenate((A_associations, B_associations))
    #scaler = MinMaxScaler(feature_range=(0,1))
    #scaler.fit(all_associations.reshape(-1,1))
    #save_scalers(SCALERS_FILEPATH, exp_num, 'second', scaler)
    
    _th = np.mean(np.abs(A_associations - B_associations))
    #_th = scaler.transform(_th.reshape(-1, 1))[0,0]
    
    biases = A_associations - B_associations
    #biases = scaler.transform(biases.reshape(-1, 1))
    QR_dict = add_quantile_ranges_to_dict({}, biases)   
    allwords_mean = np.mean(biases)

    target_associations = np.apply_along_axis(lambda x_vec: calculate_cosines_for_target_word_unscaled(x_vec, A_mtx, B_mtx), 1, target_mtx)
    
    target_biases = []
    for _assoc in target_associations:
        #_A_assoc = scaler.transform(_assoc[0].reshape(-1, 1))[0,0]
        #_B_assoc = scaler.transform(_assoc[1].reshape(-1, 1))[0,0]
        _A_assoc = _assoc[0]
        _B_assoc = _assoc[1]
        _bias = _A_assoc - _B_assoc
        target_biases.append(_bias)
    return np.array(target_biases), _th, QR_dict, allwords_mean

def run_exps_2ndorder(X_terms, Y_terms, A_terms, B_terms, exp_num):
    
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(we_model, X_terms, Y_terms, A_terms, B_terms)
    
    # A_associations, B_associations are associations for all words    
    A_associations, B_associations = calculate_cosines_for_all_words_unscaled(we_model, A_mtx, B_mtx)
    
    X_metrics, _th, QR_dict, allwords_mean = get_2ndorder_association_metric_list_for_target_list(X_mtx, A_associations, 
                                                                        B_associations, A_mtx, B_mtx, we_model, exp_num)
    Y_metrics, _th, QR_dict, allwords_mean = get_2ndorder_association_metric_list_for_target_list(Y_mtx, A_associations, 
                                                                        B_associations, A_mtx, B_mtx, we_model, exp_num)
    print (f'X_metrics: {X_metrics}')
    print (f'Y_metrics: {Y_metrics}')

    print ('mean bias to X:', np.mean(X_metrics))
    print ('mean bias to Y:', np.mean(Y_metrics))
    print ('Bias threshold:', _th)
    print(f'Quantiles: {QR_dict}')

    # Until further notice, the `order` variable is necessary here
    # due to the structure of the dictionaries where results
    # are saved.
    order = 'second'
    threshold = _th
    save_arrays(RESULTS_FILEPATH, exp_num, order, X_metrics, Y_metrics, threshold, QR_dict, allwords_mean)

def run_all_exps():
    exps = open_pickle(EXPERIMENT_DEFINITION_PATH)
    for exp_num, exp in exps.items():
        print('***********************************')
        print(f'Experiment: {exp_num}')
        X_terms = exp['X_terms']
        Y_terms = exp['Y_terms']
        A_terms = exp['A_terms']
        B_terms = exp['B_terms']
        run_exps_2ndorder(X_terms, Y_terms, A_terms, B_terms, exp_num)

if __name__ == '__main__':
    run_all_exps()
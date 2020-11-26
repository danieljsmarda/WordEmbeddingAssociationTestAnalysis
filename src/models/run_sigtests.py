import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
random.seed(5)
import sys
sys.path.append('../src')
from bias_calculation import get_matrices_from_term_lists
from utils import save_pickle, open_pickle, \
    save_experiment_arbitrary_label, \
    del_dict_entries

MODEL_NAME = input('Please type your model name, then press ENTER. This name will be \
appended to all filenames. For example, the experiment definitions will \
be loaded from ../../data/interim/<model_name>_experiment_definitions.pickle.\n\
->')
MODEL_NAME = 'glove_840B'
RESULTS_FILEPATH = f'../../data/interim/{MODEL_NAME}_association_metric_exps.pickle'
EXPERIMENT_DEFINITION_PATH = f'../../data/interim/{MODEL_NAME}_experiment_definitions.pickle'
we_model = KeyedVectors.load(f'../../data/interim/{MODEL_NAME}_norm', mmap='r')


# Fastest version, 10000 words -> 1 minute
def get_test_stat(wv_obj, X_terms, Y_terms, A_terms, B_terms):  
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(we_model, X_terms, Y_terms, A_terms, B_terms)
    cosine_sim_XA = cosine_similarity(X_mtx, A_mtx)
    cosine_sim_XB = cosine_similarity(X_mtx, B_mtx)
    mean_over_Xa = np.mean(cosine_sim_XA, axis=1)
    mean_over_Xb = np.mean(cosine_sim_XB, axis=1)
    s_for_X_words = mean_over_Xa - mean_over_Xb
    # shape is (24,) or (|X_terms|,)

    cosine_sim_YA = cosine_similarity(Y_mtx, A_mtx)
    cosine_sim_YB = cosine_similarity(Y_mtx, B_mtx)
    mean_over_Ya = np.mean(cosine_sim_YA, axis=1)
    mean_over_Yb = np.mean(cosine_sim_YB, axis=1)
    s_for_Y_words = mean_over_Ya - mean_over_Yb
    test_stat = np.mean(s_for_X_words) - np.mean(s_for_Y_words)
    return test_stat


# This cell works too. It takes twice as long as the cell above,
# but if we want to try to vectorize the outer loop, then 
# we will probably have to use this version
def calculate_association_metric_for_target_word(word_vec, A_mtx, B_mtx):
    '''Computes the association metric, s(w,A,B).
    word_vec: 1-D word vector
    A_mtx, B_mtx: 2-D word vector arrays'''
    A_dot_v = np.dot(A_mtx, word_vec)
    B_dot_v = np.dot(B_mtx, word_vec)
    A_norms = np.multiply(np.linalg.norm(A_mtx, axis=1), np.linalg.norm(word_vec))
    B_norms = np.multiply(np.linalg.norm(B_mtx, axis=1), np.linalg.norm(word_vec))
    A_cosines = np.divide(A_dot_v, A_norms)
    B_cosines = np.divide(B_dot_v, B_norms)
    return np.mean(A_cosines) - np.mean(B_cosines)

'''
for i in tqdm(range(10000)):
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(we_model, X_terms, Y_terms, A_terms, B_terms)
    X_associations = np.apply_along_axis(lambda x_vec: calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx), 1, X_mtx)
    Y_associations = np.apply_along_axis(lambda y_vec: calculate_association_metric_for_target_word(y_vec, A_mtx, B_mtx), 1, Y_mtx)
    m = np.mean(X_associations) - np.mean(Y_associations)
'''


def get_n_test_stats(wv_obj, X_terms, Y_terms, A_terms, B_terms, n_samples=100):
    sigtest_dist_1 = []
    sigtest_dist_2 = []
    sigtest_dist_3 = []
    n_targets = len(X_terms)
    n_attributes = len(A_terms)
    assert len(X_terms) == len(Y_terms)
    assert len(A_terms) == len(B_terms)
    vocab_list = list(wv_obj.vocab)
    random.seed(5)
    for i in tqdm(range(n_samples), desc=f'Calculating SigTest values for {n_samples} samples'):
        X_sample = random.sample(vocab_list, k=n_targets)
        Y_sample = random.sample(vocab_list, k=n_targets)
        sigtest_dist_1.append(get_test_stat(wv_obj, X_sample, Y_sample, A_terms, B_terms))
        sigtest_dist_2.append(get_test_stat(wv_obj, X_terms, Y_sample, A_terms, B_terms))
        sigtest_dist_3.append(get_test_stat(wv_obj, Y_terms, X_sample, A_terms, B_terms))
        #sigtest_dist_3.append(get_test_stat(wv_obj, X_sample, Y_terms, A_terms, B_terms))
    return np.array(sigtest_dist_1), np.array(sigtest_dist_2), np.array(sigtest_dist_3)


def run_all_sigtests(new_dists=False, n_samples=100):
    exps = open_pickle(EXPERIMENT_DEFINITION_PATH)
    results_dict = open_pickle(RESULTS_FILEPATH)
    # `order` variable only necessary for saving files.
    order = 'second'
    for exp_num, exp in exps.items():
        print('******************************')
        print(f'Experiment: {exp_num}')
        X_terms = exp['X_terms']
        Y_terms = exp['Y_terms']
        A_terms = exp['A_terms']
        B_terms = exp['B_terms']
        
        comparison_statistic = get_test_stat(we_model, X_terms, Y_terms, A_terms, B_terms)
        if new_dists:
            dist_1, dist_2, dist_3 = get_n_test_stats(we_model, X_terms, Y_terms, A_terms, B_terms, n_samples=n_samples)
        else:
            dist_1, dist_2, dist_3 = [results_dict[exp_num][order][f'sigtest_dist_{n}'] for n in [1,2,3]]
        p_value = norm.cdf(comparison_statistic, loc=np.mean(dist_1), scale=np.std(dist_1))
  
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order,
                                        'sigtest_dist_1', dist_1)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order,
                                        'sigtest_dist_2', dist_2)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order,
                                        'sigtest_dist_3', dist_3)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order,
                                        'test_statistic', comparison_statistic)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order,
                                        'ST1_p-value', p_value)


if __name__ == '__main__':
    n_samples = None
    rerun = input('Do you want to calculate new samples? Caution:\
This will overwrite previously-calculated samples. (y/n)\n->')
    if rerun not in ['y','n']:
        print('Invalid answer. Please rerun and enter either "y" or "n".')
    elif rerun=='y':
        n_samples = int(input('How many samples?\n->'))
    run_all_sigtests(rerun, n_samples)
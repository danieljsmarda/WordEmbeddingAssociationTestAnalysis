import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random
random.seed(5)
from bias_calculation import get_matrices_from_term_lists
from utils import open_pickle, \
    save_experiment_arbitrary_label

MODEL_NAME = input('Please type your model name, then press ENTER. This name will be \
appended to all filenames. For example, the experiment definitions will \
be loaded from ../../data/interim/<model_name>_experiment_definitions.pickle.\n\
->')
RESULTS_FILEPATH = f'../../data/interim/{MODEL_NAME}_association_metric_exps.pickle'
EXPERIMENT_DEFINITION_PATH = f'../../data/interim/{MODEL_NAME}_experiment_definitions.pickle'
we_model = KeyedVectors.load(f'../../data/interim/{MODEL_NAME}_norm', mmap='r')


# Fastest version, 10000 samples -> 1 minute per experiment
def get_test_stat(we_model, X_terms, Y_terms, A_terms, B_terms):  
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

# Fastest version, 10000 samples -> 1 minute per experiment
def get_test_stat_onesided(we_model, Z_terms, A_terms, B_terms):  
    [Z_mtx, _, A_mtx, B_mtx] = get_matrices_from_term_lists(we_model, Z_terms, Z_terms, A_terms, B_terms)
    cosine_sim_ZA = cosine_similarity(Z_mtx, A_mtx)
    cosine_sim_ZB = cosine_similarity(Z_mtx, B_mtx)
    mean_over_Za = np.mean(cosine_sim_ZA, axis=1)
    mean_over_Zb = np.mean(cosine_sim_ZB, axis=1)
    s_for_Z_words = mean_over_Za - mean_over_Zb
    
    test_stat = np.mean(s_for_Z_words)
    return test_stat

def get_n_test_stats(wv_obj, X_terms, Y_terms, A_terms, B_terms, n_samples=100):
    sigtest_dist_1 = []
    sigtest_dist_2 = []
    sigtest_dist_3 = []
    n_targets = len(X_terms)
    assert len(X_terms) == len(Y_terms), "Target list lengths are unequal."
    assert len(A_terms) == len(B_terms), "Attribute list lengths are unequal."
    vocab_list = list(wv_obj.vocab)
    random.seed(5)
    for i in tqdm(range(n_samples), desc=f'Calculating SigTest values for {n_samples} samples'):
        X_sample = random.sample(vocab_list, k=n_targets)
        Y_sample = random.sample(vocab_list, k=n_targets)
        sigtest_dist_1.append(get_test_stat(wv_obj, X_sample, Y_sample, A_terms, B_terms))
        _dist_onesided = get_test_stat_onesided(wv_obj, X_sample, A_terms, B_terms)
        sigtest_dist_2.append(_dist_onesided)
        sigtest_dist_3.append(_dist_onesided)
        # If the following line is used, we expect ST2 and 3 means to be roughly
        # the same on visualizations. If the line above is used, we expect them to be
        # roughly mirror images.
        #sigtest_dist_3.append(get_test_stat(wv_obj, X_sample, Y_terms, A_terms, B_terms))
    return np.array(sigtest_dist_1), np.array(sigtest_dist_2), np.array(sigtest_dist_3)

def save_sigtest_pvalues(dists, exp_num, test_statistics):
    pvalues = {}
    for i in range(3):
        dist = dists[i]
        test_statistic = test_statistics[i]
        
        loc = np.mean(dist)
        scale = np.std(dist, ddof=1)
        p = norm.cdf(test_statistic, loc=loc, scale=scale)
        pvalues[f'sigtest_{i+1}'] = p
    save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, 'second', 'sigtest_pvalues', pvalues)

def save_errorbar_values(dists, exp_num):
    '''This function calculates the error bar values that will be plotted.
    These values are the size of the errorbar in each direction.
    They are meant to be passed to the matplotlib function 
    ax.errorbar.
    dists -- list of 3 sigtest dists
    exp_num -- int
    Returns None, but saves the numbers to the results file.'''

    errors = defaultdict(dict)
    intervals = ['95', '99', '99.9']
    for interval_size in intervals:
        for i in range(3):
            dist = dists[i]
            loc = np.mean(dist)
            scale = np.std(dist, ddof=1)
            interval_pct = float(interval_size)/100
            #err = loc - norm.ppf((1-interval_pct)/2, loc=loc, scale=scale)
            err = loc - norm.ppf((1-interval_pct)/2, loc=loc, scale=scale)
            errors[f'sigtest_{i+1}'][interval_size] = err
    # The 'second' argument is necessary for saving
    save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, 'second', 'errorbar_values', errors)

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
        
        test_statistic_dist_1 = get_test_stat(we_model, X_terms, Y_terms, A_terms, B_terms)
        test_statistic_dist_2 = get_test_stat_onesided(we_model, X_terms, A_terms, B_terms)
        test_statistic_dist_3 = get_test_stat_onesided(we_model, Y_terms, A_terms, B_terms)
        
        if new_dists:
            dist_1, dist_2, dist_3 = get_n_test_stats(we_model, X_terms, Y_terms, A_terms, B_terms, n_samples=n_samples)
        else:
            dist_1, dist_2, dist_3 = [results_dict[exp_num][order][f'sigtest_dist_{n}'] for n in [1,2,3]]
        #p_value = norm.cdf(test_statistic, loc=np.mean(dist_1), scale=np.std(dist_1))
  
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'sigtest_dist_1', dist_1)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'sigtest_dist_2', dist_2)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'sigtest_dist_3', dist_3)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'test_statistic_dist_1', test_statistic_dist_1)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'test_statistic_dist_2', test_statistic_dist_2)
        save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'test_statistic_dist_3', test_statistic_dist_3)
        #save_experiment_arbitrary_label(RESULTS_FILEPATH, exp_num, order, 'ST1_p-value', p_value)
        save_errorbar_values([dist_1, dist_2, dist_3], exp_num)
        save_sigtest_pvalues([dist_1, dist_2, dist_3], exp_num, [test_statistic_dist_1, test_statistic_dist_2, test_statistic_dist_3])


if __name__ == '__main__':
    n_samples = 0
    rerun = input('Do you want to calculate new samples? Caution: \
Unless you typed a different model name, this will overwrite previously-calculated samples. (y/n)\n->')
    if rerun not in ['y','n']:
        print('Invalid answer. Please rerun and enter either "y" or "n". If "n", metrics will\
be calculated on existing samples.')
    elif rerun=='y':
        n_samples = int(input('How many samples?\n->'))
    rerun_bool = True if rerun=='y' else False
    run_all_sigtests(rerun_bool, n_samples)


##### Old/Extraneous Code #####
# This function has the same purpose as get_test_stat. 
# Between this function and get_test stat, get_test_stat is faster
# and so get_test_stat is used in the rest of the code. However,
# an even faster version could be implemented by further vectorizing
# these operations with an additional dimension added to the arrays
# corresponding to n_samples.
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
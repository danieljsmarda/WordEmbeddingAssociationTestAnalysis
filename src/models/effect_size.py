from scipy.special import comb as num_combinations
from itertools import combinations
from functools import lru_cache
from tqdm import tqdm
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy as sp
import operator

def word_set_to_mtx(wv_obj, word_set):
    '''Converts set of string words into a 2-D numpy array of word vectors from the word-vector object.'''
    return np.vstack(tuple(wv_obj[word] for word in word_set))

def get_matrices_from_term_lists(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    '''Uses wv_obj to convert lists of words to arrays of word vectors.
    Returns: list of matrices containing the corresponding vectors for 
    the words in X_terms, Y_terms, A_terms, and B_terms.'''
    X_getter = operator.itemgetter(*X_terms)
    Y_getter = operator.itemgetter(*Y_terms)
    A_getter = operator.itemgetter(*A_terms)
    B_getter = operator.itemgetter(*B_terms)
    return [np.vstack(getter(wv_obj.wv)) for getter in [X_getter, Y_getter, A_getter, B_getter]]
    

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

def get_2ndorder_association_metric_list_for_target_list(target_list, A_terms, B_terms, we_model):
    global ORDER
    ORDER = 'second'
    
    [X_mtx, _, A_mtx, B_mtx] = get_matrices_from_term_lists(we_model, target_list, target_list, A_terms, B_terms)
    associations = np.apply_along_axis(lambda x_vec: calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx), 1, X_mtx)
    
    all_words_mtx = we_model.wv.vectors
    all_associations = np.array([])
    for x_vec in tqdm(all_words_mtx, desc='Getting association metric for all words'):
        metric = calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx)
        all_associations = np.append(all_associations, metric)
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(all_associations.reshape(-1,1)) # Reshape is for a single feature, NOT for a single sample
    transformed = scaler.transform(associations.reshape(-1,1))
    return transformed.reshape(len(transformed))

def calculate_effect_size(X_mtx, Y_mtx, A_mtx, B_mtx):
    '''Computes the effect size.
    X_mtx, Y_mtx, A_mtx, B_mtx: 2-D word vector arrays.'''
    x_associations = np.apply_along_axis(lambda x_vec: calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx), 1, X_mtx)
    y_associations = np.apply_along_axis(lambda y_vec: calculate_association_metric_for_target_word(y_vec, A_mtx, B_mtx), 1, Y_mtx)
    X_union_Y = np.vstack((X_mtx, Y_mtx))
    all_associations = np.apply_along_axis(lambda w_vec: calculate_association_metric_for_target_word(w_vec, A_mtx, B_mtx), 1, X_union_Y)
    return (np.mean(x_associations) - np.mean(y_associations))/np.std(all_associations, ddof=1)

def produce_2ndorder_effect_size(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    '''Highest-level function, from word-vector object to output effect size.'''
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(wv_obj, X_terms, Y_terms, A_terms, B_terms)
    return calculate_effect_size(X_mtx, Y_mtx, A_mtx, B_mtx)

########## p-values ###########

def produce_test_statistic(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    # THIS IS A DEPRECATED FUNCTION
    '''Calculates test statistic s(X,Y,A,B).'''
    [X_mtx, Y_mtx, A_mtx, B_mtx] = get_matrices_from_term_lists(wv_obj, X_terms, Y_terms, A_terms, B_terms)
    x_associations = np.apply_along_axis(lambda x_vec: calculate_association_metric_for_target_word(x_vec, A_mtx, B_mtx), 1, X_mtx)
    y_associations = np.apply_along_axis(lambda y_vec: calculate_association_metric_for_target_word(y_vec, A_mtx, B_mtx), 1, Y_mtx)
    return np.sum(x_associations) - np.sum(y_associations)

def get_complements(x_union_y):
    '''Generator function that yields pairs of equal-size disjoint subsets
    of x_union_y.
    x_union_y should a set type.'''
    for seq in combinations(x_union_y, len(x_union_y)//2):
        complement = frozenset(x_union_y.difference(seq))
        yield (seq, complement)

def produce_2ndorder_p_value(wv_obj, X_terms, Y_terms, A_terms, B_terms):
    # THIS IS A DEPRECATED FUNCTION
    '''Generates the p-value for a set of terms with the word-vector object.
    High-level function; this function should be directly imported into 
    notebooks for experimentation.'''
    x_union_y = set(X_terms).union(set(Y_terms))
    total_terms = len(x_union_y)
    comparison_statistic = produce_test_statistic(wv_obj, X_terms, Y_terms, A_terms, B_terms)
    dist = np.array([])
    for (X_i_terms, Y_i_terms) in tqdm(get_complements(x_union_y), total=num_combinations(total_terms, total_terms/2)):
        test_statistic = produce_test_statistic(wv_obj, X_i_terms, Y_i_terms, A_terms, B_terms)
        dist = np.append(dist, test_statistic)
    return 1 - sp.stats.norm.cdf(comparison_statistic, loc=np.mean(dist), scale=np.std(dist, ddof=1))

############## 1st-order #####################
@lru_cache(maxsize=None)
def get_expSG_vecs(words, we_model, E_ctx_vec_tup, E_wrd_vec_tup):
    E_ctx_vec = np.array(E_ctx_vec_tup)
    E_wrd_vec = np.array(E_wrd_vec_tup)
    expSG_vecs = {}
    for word in words:
        _idx = we_model.wv.vocab[word].index
        _vec = we_model.wv.vectors[_idx]
        
        # explicit SkipGram
        expSG_vecs[word] = sp.special.expit(np.dot(we_model.trainables.syn1neg, _vec))
        expSG_vecs[word] /= np.sqrt(E_ctx_vec * E_wrd_vec[_idx])
    
    return expSG_vecs

@lru_cache(maxsize=None)
def get_expSG_1storder_relation(word_from, words_to, we_model, E_ctx_vec_tup, E_wrd_vec_tup):

    expSG_vec = get_expSG_vecs(tuple([word_from]), we_model, E_ctx_vec_tup, E_wrd_vec_tup)[word_from]
    
    relations={}
    for word_to in words_to:
        if word_to in we_model.wv.vocab:
            _idx = we_model.wv.vocab[word_to].index
            relations[word_to] = expSG_vec[_idx]
    
    return relations

@lru_cache(maxsize=None)
def get_1storder_association_metric(word, A_terms, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup):

    A_relations = get_expSG_1storder_relation(word, A_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup)
    B_relations = get_expSG_1storder_relation(word, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup)
    return mean(A_relations.values()) - mean(B_relations.values())

@lru_cache(maxsize=None)
def produce_1storder_effect_size_unnormalized(X_terms, Y_terms, A_terms, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup):
    x_associations = np.array([])
    y_associations = np.array([])
    for (x,y) in zip(X_terms, Y_terms):
        x_association = get_1storder_association_metric(x, A_terms, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup)
        y_association = get_1storder_association_metric(y, A_terms, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup)
        x_associations = np.append(x_associations, x_association)
        y_associations = np.append(y_associations, y_association)
    all_associations = np.append(x_associations, y_associations)
    return (np.mean(x_associations) - np.mean(y_associations))/np.std(all_associations, ddof=1)

@lru_cache(maxsize=None)
def produce_1storder_test_statistic(we_model, X_terms, Y_terms, A_terms, B_terms, E_ctx_vec_tup, E_wrd_vec_tup):
    x_associations = np.array([])
    y_associations = np.array([])
    for (x,y) in zip(X_terms, Y_terms):
        x_association = get_1storder_association_metric(x, A_terms, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup)
        y_association = get_1storder_association_metric(y, A_terms, B_terms, we_model, E_ctx_vec_tup, E_wrd_vec_tup)
        x_associations = np.append(x_associations, x_association)
        y_associations = np.append(y_associations, y_association)
    return np.sum(x_associations) - np.sum(y_associations)

def produce_1storder_p_value(we_model, X_terms, Y_terms, A_terms, B_terms, E_ctx_vec, E_wrd_vec):
    '''Generates the p-value for a set of terms with the word-vector object.
    High-level function; this function should be directly imported into 
    notebooks for experimentation.'''
    x_union_y = set(X_terms).union(set(Y_terms))
    total_terms = len(x_union_y)
    [X_terms, Y_terms, A_terms, B_terms] = [tuple(x) for x in [X_terms, Y_terms, A_terms, B_terms]]
    E_ctx_vec_tup = tuple(E_ctx_vec)
    E_wrd_vec_tup = tuple(E_wrd_vec)
    comparison_statistic = produce_1storder_test_statistic(we_model, X_terms, Y_terms, A_terms, B_terms, E_ctx_vec_tup, E_wrd_vec_tup)
    dist = np.array([])
    for (X_i_terms, Y_i_terms) in tqdm(get_complements(x_union_y), total=num_combinations(total_terms, total_terms/2)):
        test_statistic = produce_1storder_test_statistic(we_model, X_i_terms, Y_i_terms, A_terms, B_terms, E_ctx_vec_tup, E_wrd_vec_tup)
        dist = np.append(dist, test_statistic)
    return 1 - sp.stats.norm.cdf(comparison_statistic, loc=np.mean(dist), scale=np.std(dist, ddof=1))
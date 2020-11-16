from scipy.spatial.distance import cosine as cosine_distance
from scipy.special import comb as num_combinations
from itertools import combinations
from functools import lru_cache
from tqdm import tqdm
from statistics import mean
import numpy as np
import scipy as sp

def get_complements(x_union_y):
    '''Generator function that yields pairs of equal-size disjoint subsets
    of x_union_y.
    x_union_y should a set type.'''
    for seq in combinations(x_union_y, len(x_union_y)//2):
        complement = frozenset(x_union_y.difference(seq))
        yield (seq, complement)

def get_expSG_vecs_no_cache(words, we_model, E_ctx_vec, E_wrd_vec):
    E_ctx_vec = np.array(E_ctx_vec)
    E_wrd_vec = np.array(E_wrd_vec)
    expSG_vecs = {}
    for word in words:
        _idx = we_model.wv.vocab[word].index
        _vec = we_model.wv.vectors[_idx]
        
        # explicit SkipGram
        expSG_vecs[word] = sp.special.expit(np.dot(we_model.trainables.syn1neg, _vec))
        expSG_vecs[word] /= np.sqrt(E_ctx_vec * E_wrd_vec[_idx])
    
    return expSG_vecs


def get_expSG_1storder_relation_no_cache(word_from, words_to, we_model, E_ctx_vec, E_wrd_vec):
    expSG_vec = get_expSG_vecs_no_cache(tuple([word_from]), we_model, E_ctx_vec, E_wrd_vec)[word_from]
    
    relations={}
    for word_to in words_to:
        if word_to in we_model.wv.vocab:
            _idx = we_model.wv.vocab[word_to].index
            relations[word_to] = expSG_vec[_idx]
    
    return relations

def get_1storder_association_metric_no_cache(word, A_terms, B_terms, we_model, E_ctx_vec, E_wrd_vec):
    A_relations = get_expSG_1storder_relation_no_cache(word, A_terms, we_model, E_ctx_vec, E_wrd_vec)
    B_relations = get_expSG_1storder_relation_no_cache(word, B_terms, we_model, E_ctx_vec, E_wrd_vec)
    return mean(A_relations.values()) - mean(B_relations.values())

def produce_1storder_effect_size_unnormalized_no_cache(X_terms, Y_terms, A_terms, B_terms, we_model, E_ctx_vec, E_wrd_vec):
    for (x,y) in zip(X_terms, Y_terms):
        x_association = get_1storder_association_metric_no_cache(x, A_terms, B_terms, we_model, E_ctx_vec, E_wrd_vec)
        y_association = get_1storder_association_metric_no_cache(y, A_terms, B_terms, we_model, E_ctx_vec, E_wrd_vec)
        x_associations = np.append(x_associations, x_association)
        y_associations = np.append(y_associations, y_association)
    all_associations = np.append(x_associations, y_associations)
    return (np.mean(x_associations) - np.mean(y_associations))/np.std(all_associations, ddof=1)

def produce_1storder_test_statistic_no_cache(we_model, X_terms, Y_terms, A_terms, B_terms, E_ctx_vec, E_wrd_vec):
    for (x,y) in zip(X_terms, Y_terms):
        x_association = get_1storder_association_metric_no_cache(x, A_terms, B_terms, we_model, E_ctx_vec, E_wrd_vec)
        y_association = get_1storder_association_metric_no_cache(y, A_terms, B_terms, we_model, E_ctx_vec, E_wrd_vec)
        x_associations = np.append(x_associations, x_association)
        y_associations = np.append(y_associations, y_association)
    return np.sum(x_associations) - np.sum(y_associations)

def produce_1storder_p_value_no_cache(we_model, X_terms, Y_terms, A_terms, B_terms, E_ctx_vec, E_wrd_vec):
    '''Generates the p-value for a set of terms with the word-vector object.
    High-level function; this function should be directly imported into 
    notebooks for experimentation.'''
    x_union_y = set(X_terms).union(set(Y_terms))
    total_terms = len(x_union_y)
    [X_terms, Y_terms, A_terms, B_terms] = [tuple(x) for x in [X_terms, Y_terms, A_terms, B_terms]]
    E_ctx_vec = tuple(E_ctx_vec)
    E_wrd_vec = tuple(E_wrd_vec)
    comparison_statistic = produce_1storder_test_statistic_no_cache(we_model, X_terms, Y_terms, A_terms, B_terms, E_ctx_vec, E_wrd_vec)
    dist = np.array([])
    for (X_i_terms, Y_i_terms) in tqdm(get_complements(x_union_y), total=num_combinations(total_terms, total_terms/2)):
        test_statistic = produce_1storder_test_statistic_no_cache(we_model, X_i_terms, Y_i_terms, A_terms, B_terms, E_ctx_vec, E_wrd_vec)
        dist = np.append(dist, test_statistic)
    return 1 - sp.stats.norm.cdf(comparison_statistic, loc=np.mean(dist), scale=np.std(dist, ddof=1))
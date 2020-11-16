from itertools import combinations
from tqdm import tqdm
import numpy as np
import scipy as sp
from sklearn.preprocessing import MinMaxScaler

def get_complements(x_union_y):
    '''Generator function that yields pairs of equal-size disjoint subsets
    of x_union_y.
    x_union_y should a set type.'''
    for seq in combinations(x_union_y, len(x_union_y)//2):
        complement = frozenset(x_union_y.difference(seq))
        yield (seq, complement)

def get_expSG_1storder_relation_no_cache_NEW(word_from, words_to, we_model):
    ctx_vecs = []
    for _word in words_to:
        _idx = we_model.wv.vocab[_word].index
        ctx_vecs.append(we_model.trainables.syn1neg[_idx])
    ctx_vecs = np.array(ctx_vecs)    
    
    _vec = we_model.wv[word_from]
    relations = sp.special.expit(np.dot(ctx_vecs, _vec))
    
    return relations

def get_expSG_1storder_relation_no_cache_NEW_ALLWORDS(words_to, we_model):
    ctx_vecs = []
    for _word in words_to:
        _idx = we_model.wv.vocab[_word].index
        ctx_vecs.append(we_model.trainables.syn1neg[_idx])
    ctx_vecs = np.array(ctx_vecs)    
    
    _vecs = we_model.wv.vectors
    relations = sp.special.expit(np.dot(_vecs, ctx_vecs.T))
    
    return relations

def get_1storder_association_metric_fast(word, A_terms, B_terms, we_model):
    A_relations = get_expSG_1storder_relation_no_cache_NEW(word, A_terms, we_model)
    B_relations = get_expSG_1storder_relation_no_cache_NEW(word, B_terms, we_model)
    return np.mean(A_relations) - np.mean(B_relations)

def get_all_relations_1storder(A_terms, B_terms, we_model):
    A_relations=get_expSG_1storder_relation_no_cache_NEW_ALLWORDS(A_terms, we_model)
    B_relations=get_expSG_1storder_relation_no_cache_NEW_ALLWORDS(B_terms, we_model)
    all_associations = np.mean(A_relations, axis=1) - np.mean(B_relations, axis=1)
    return all_associations

def get_1storder_association_metric_list_for_target_list(target_list, A_terms, B_terms, we_model):
    global ORDER
    ORDER = 'first'
    
    associations = np.array([])
    for word in tqdm(target_list):
        association = get_1storder_association_metric_fast(word, A_terms, B_terms, we_model)
        associations = np.append(associations, association)
    
    scaler = MinMaxScaler(feature_range=(-1,1))
    all_associations = get_all_relations_1storder(A_terms, B_terms, we_model)
    scaler.fit(all_associations.reshape(-1,1)) # Reshape is for a single feature, NOT for a single sample
    transformed = scaler.transform(associations.reshape(-1,1))
    return transformed.reshape(len(transformed))
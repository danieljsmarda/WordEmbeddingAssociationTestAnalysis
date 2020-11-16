import pickle
import numpy as np
from collections import defaultdict

def save_pickle(obj, FILEPATH):
    f = open(FILEPATH, 'wb')
    pickle.dump(obj, f)
    f.close()

def open_pickle(FILEPATH):
    f = open(FILEPATH, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj

def save_arrays(FILEPATH, exp_num, order, X_metrics, Y_metrics, threshold,
    pct_5, pct_95, A_biases, lower_bound, upper_bound):
    results_dict = open_pickle(FILEPATH)
    results_dict[exp_num] = results_dict.get(exp_num, defaultdict(dict))
    order_dict = results_dict[exp_num].get(order, {})
    order_dict['X_array'] = X_metrics
    order_dict['Y_array'] = Y_metrics
    order_dict['X_mean'] = np.mean(X_metrics)
    order_dict['Y_mean'] = np.mean(Y_metrics)
    order_dict['threshold'] = threshold
    #order_dict['pct_5'] = pct_5
    #order_dict['pct_95'] = pct_95
    #order_dict['A_biases'] = A_biases
    #order_dict['lower_bound'] = lower_bound
    #order_dict['upper_bound'] = upper_bound
    results_dict[exp_num][order] = order_dict
    save_pickle(results_dict, FILEPATH)
    print(f"Results array successfully saved to file {FILEPATH} under\
 keys [{exp_num}][{order}]")

def save_experiment_arbitrary_label(filepath, exp_num, order, label, data, display=None):
    results_dict = open_pickle(filepath)
    results_dict[exp_num] = results_dict.get(exp_num, defaultdict(dict))
    order_dict = results_dict[exp_num].get(order, {})
    order_dict[label] = data
    results_dict[exp_num][order] = order_dict
    save_pickle(results_dict, filepath)
    if display == 'all':
        print(f'FULL RESULTS DICT FOR EXP {exp_num}', results_dict[exp_num])
    elif display == 'some':
        print(f'SPECIFIC RESULTS FOR EXP {exp_num}, LABEL "{label}": \
        {results_dict[exp_num][order][label]}')
    print(f"Results array successfully saved to file {filepath} under\
    keys [{exp_num}][{order}][{label}]")

def save_scalers(filepath, exp_num, order, scaler): 
    results_dict = open_pickle(filepath)
    results_dict[exp_num] = results_dict.get(exp_num, defaultdict(dict))
    results_dict[exp_num][order] = scaler
    save_pickle(results_dict, filepath)

def save_array_old(FILEPATH, arr, exp_num, order, list_name):
    results_dict = open_pickle(FILEPATH)
    exp_name = str(order)+'_order_'+list_name
    results_dict[exp_num][exp_name] = arr
    save_pickle(results_dict, FILEPATH)
    print(f"Results array successfully saved to file {FILEPATH} under\
 keys [{exp_num}]['{exp_name}']")

def filter_terms_not_in_wemodel(we_model, X_terms, Y_terms):
    term_list_names = ['first_list', 'second_list']
    term_lists = [X_terms, Y_terms]
    for i in range(len(term_lists)):
        lst = term_lists[i]
        name = term_list_names[i]
        unknown_words = [w for w in lst if w not in we_model.wv]
        print(f'The following terms were removed from the list {name} because they were not found in the we_model: {unknown_words}')
    X_terms_filtered = [x for x in X_terms if x in we_model.wv]
    Y_terms_filtered = [y for y in Y_terms if y in we_model.wv]
    diff = abs(len(X_terms_filtered) - len(Y_terms_filtered))
    if len(X_terms_filtered) > len(Y_terms_filtered):
        print(f'The following terms were removed from the first list to balance the length of the lists: {X_terms_filtered[:diff]}')
        X_terms_filtered = X_terms_filtered[diff:]
    elif len(Y_terms_filtered) > len(X_terms_filtered):
        print(f'The following terms were removed from the second list to balance the length of the lists: {Y_terms_filtered[:diff]}')
        Y_terms_filtered = Y_terms_filtered[diff:]
    return (X_terms_filtered, Y_terms_filtered)


import pickle
import numpy as np
from collections import defaultdict
import os

def save_pickle(obj, filepath):
    '''Save obj to the specified filepath.'''
    f = open(filepath, 'wb')
    pickle.dump(obj, f)
    f.close()

def open_pickle(filepath):
    '''Get obj from the specified filepath.'''
    if os.path.exists(filepath):
        f = open(filepath, 'rb')
        obj = pickle.load(f)
        f.close()
    else:
        obj = None
    return obj

def save_arrays(FILEPATH, exp_num, order, X_metrics, Y_metrics, threshold, QR_dict, allwords_mean, all_biases):
    '''Primary save function for the results of the single-word experiments.'''
    results_dict = open_pickle(FILEPATH)
    if results_dict == None:
        results_dict = {}
    results_dict[exp_num] = results_dict.get(exp_num, defaultdict(dict))
    order_dict = results_dict[exp_num].get(order, {})
    order_dict['X_array'] = X_metrics
    order_dict['Y_array'] = Y_metrics
    order_dict['X_mean'] = np.mean(X_metrics)
    order_dict['Y_mean'] = np.mean(Y_metrics)
    order_dict['threshold'] = threshold
    order_dict['QR_dict'] = QR_dict
    order_dict['allwords_mean'] = allwords_mean
    order_dict['all_biases'] = all_biases
    results_dict[exp_num][order] = order_dict
    save_pickle(results_dict, FILEPATH)
    print(f"Results successfully saved to file {FILEPATH} under\
 keys [{exp_num}][\'{order}\']")

def del_dict_entries(dct, filepath, keys=[]):
    '''This function is a helper function for developers, not used
    elsewhere in the code. It was added because data is currently
    only appended to the data dictionaries. This function
    can be called to delete entries from the results dictionary.'''
    for key in keys:
        if key in dct[1]['second'].keys():
            print(f'The key \'{key}\' has been successfully deleted.')
        else:
            print(f'Key \'{key}\' not found in dictionary.')
        for exp_num in range(1,11):
            dct[exp_num]['second'].pop(key, None)
    save_pickle(dct, filepath)
    print(f'The new dictionary has been successfully saved to {filepath}')

def save_experiment_arbitrary_label(filepath, exp_num, order, label, data, display=None):
    '''This function serves a similar purpose as save_arrays. However,
    save_arrays is specific to saving the results of the single-word experiments.
    This function allows saving of any result.'''
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
    print(f"Results successfully saved to file {filepath} under\
keys [{exp_num}][\'{order}\'][\'{label}\']")

def save_scalers(filepath, exp_num, order, scaler): 
    '''This function was specifically written to save MinMaxScalers
    used in the single-word experiments.'''
    results_dict = open_pickle(filepath)
    results_dict[exp_num] = results_dict.get(exp_num, defaultdict(dict))
    results_dict[exp_num][order] = scaler
    save_pickle(results_dict, filepath)
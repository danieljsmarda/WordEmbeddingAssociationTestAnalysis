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
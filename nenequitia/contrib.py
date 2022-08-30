import copy


def get_manuscripts_and_lang_kfolds(dataframe, k=0, per_k=2, force_test=None):
    all_data = {}
    for lang, mss in dataframe.set_index(['lang', 'manuscript']).sort_index().index.unique():
        if lang not in all_data:
            all_data[lang] = []
        if force_test and mss in force_test:
            continue
        all_data[lang].append(mss)

    train, dev, test = [], [], []
    if force_test:
        test.extend(force_test)
    local_data = copy.deepcopy(all_data)
    for lang in all_data:

        for i in range(per_k):
            dev.append(local_data[lang].pop(k*per_k+i))

        for i in range(per_k):
            test.append(local_data[lang].pop(k*per_k+i))

        train.extend(local_data[lang])
    return (
        dataframe.loc[dataframe.manuscript.isin(train)],
        dataframe.loc[dataframe.manuscript.isin(dev)],
        dataframe.loc[dataframe.manuscript.isin(test)]
    )

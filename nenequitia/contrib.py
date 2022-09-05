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


if __name__ == "__main__":
    from pandas import read_hdf

    df = read_hdf("../texts.hdf5", key="df", index_col=0)
    rows = [

    ]
    df["bin"] = ""
    df.loc[df.CER < 10, "bin"] = "Good"
    df.loc[df.CER.between(10, 25, inclusive="left"), "bin"] = "Acceptable"
    df.loc[df.CER.between(25, 50, inclusive="left"), "bin"] = "Bad"
    df.loc[df.CER >= 50, "bin"] = "Very bad"

    for (lr, dropout) in [(5e-4, .1)]:
        for i in range(5):
            print(i)
            train, dev, test = get_manuscripts_and_lang_kfolds(
                df,
                k=i, per_k=2,
                force_test=["SBB_PK_Hdschr25"]
            )
            rows.append({
                "dev fro": (", ".join(dev[dev.lang == "fro"].manuscript.unique().tolist())).replace("_", " "),
                "dev lat": (", ".join(dev[dev.lang == "lat"].manuscript.unique().tolist())).replace("_", " "),
                "test fro": (", ".join(test[test.lang == "fro"].manuscript.unique().tolist())).replace("_", " "),
                "test lat": (", ".join([
                    m for m in test[test.lang == "lat"].manuscript.unique().tolist()
                    if not m.startswith("SBB")
                ])).replace("_", " ")}
                        )
            print(rows[-1])
    from pandas import DataFrame
    print(DataFrame(rows).transpose().to_latex(index=True))

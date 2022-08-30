#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LinearRegression, RANSACRegressor, RidgeCV, SGDRegressor


# In[2]:


df = pd.read_hdf("features.hdf5", key="df", index_col=0)
df.drop(['manuscript', 'page_id', "line_id", "transcription"], axis=1, inplace=True)
df["K"] = 0
print(sorted(df.bin.unique()))

# In[3]:


df.head()


# In[4]:


df.bin.unique()


# # Generate K-Folds class

# In[5]:


KS = 10

#kf = KFold(n_splits=KS, shuffle = True, random_state = 2)

for unique_bin in df.bin.unique():
    ids = list(np.array_split(np.array(df[df.bin == unique_bin].index), KS))
    for k, k_ids in enumerate(ids):
        df.loc[k_ids, "K"] = k

df.tail()


print(f"Memory {df.info(memory_usage='deep')}")
#df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)
#print(f"Memory {df.info(memory_usage='deep')}")
# In[6]:


#df.to_hdf("k-fold.hdf", key="kfold")


# In[7]:


def get_kfold_train_test(
    dataframe: pd.DataFrame,
    k=0
):
    # Right now only deal with train and test
    ks = list(range(10))
    test = ks[k]
    train = ks[:k]+ks[k+1:]
    dev = train.pop(0)
    return (
        dataframe[dataframe['K'].isin(train)].index.tolist()[:1024],
        dataframe[dataframe['K'] == dev].index.tolist()[:1024],
        dataframe[dataframe['K'] == test].index.tolist()
    )


# In[8]:


#get_kfold_train_test(df)


# In[9]:


import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import tqdm
import json
from typing import List, Optional
import random

        
def var(X):
    return torch.from_numpy(X).cuda()


class NgramModel(nn.Module):
    def __init__(self, features, output_dim, device="cuda:0", ):
        super(NgramModel, self).__init__()
        self.out = output_dim
        self.features = features or ()
        self.inp = len(self.features)
        self.net = nn.Sequential(
            nn.Linear(self.inp, 128),
            nn.Dropout(.1),
            nn.Linear(128, self.out)
        )
        self.to(device)
        
    def get_batches(
        self,
        dataframe,
        indexes: Optional[List[int]],
        has_truth: bool = False,
        batch_size=256
    ):
        if has_truth:
            random.shuffle(indexes)
        samples = len(indexes)
        
        for batch_start in range(0, samples, batch_size):
            batch_end = min(batch_start+batch_size, samples)
            loc_indexes = indexes[batch_start:batch_end]
            if has_truth:
                yield (
                    var(dataframe.loc[loc_indexes, self.features].fillna(.0).to_numpy()),
                    var(dataframe.loc[loc_indexes, ["bin"]].to_numpy(dtype="l")).squeeze(),
                )
            else:
                yield (
                    var(dataframe.loc[loc_indexes, self.features].fillna(.0).to_numpy()),
                    None
                )
        
    def fit(
        self, 
        dataframe: pd.DataFrame,
        train_indexes: List[int], dev_indexes: List[int],
        epochs=1000, max_bad_epochs=20, batch_size=512, 
        lr=5e-3, delta=.005, use_loss: bool = True
    ):
        with torch.cuda.amp.autocast():
            criterion = torch.nn.CrossEntropyLoss()
            dev_loss = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
            best = float("-inf")
            if use_loss:
                best = float("inf")
                
            bad_epochs = 0
            best_params = self.state_dict()
            
            nb_batches = df.shape[0] // batch_size
            for epoch in (pbar := tqdm.tqdm(range(epochs))):
                            
                for batch_idx, (xs, ys) in tqdm.tqdm(enumerate(
                    self.get_batches(dataframe, train_indexes, batch_size=batch_size, has_truth=True)
                ), leave=True):
                    outputs = self.net(xs)
                    criterion(outputs, ys).backward()

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

                self.eval()
                predicted = []
                total_ys = []
                epoch_loss = torch.tensor(.0).cpu().item()

                for batch_idx, (xs, ys) in tqdm.tqdm(enumerate(
                    self.get_batches(dataframe, dev_indexes, batch_size=batch_size, has_truth=True)
                ), leave=True):
                    outputs = self.net(xs)
                    loss = dev_loss(outputs, ys)
                    epoch_loss += loss.item()

                    predicted.extend(outputs.argmax(dim=-1).cpu().flatten().tolist())
                    total_ys.extend(ys.tolist())

                acc = accuracy_score(predicted, total_ys)
                epoch_loss = epoch_loss / (batch_idx + 1)
                self.train()

                factor = 100

                if use_loss:
                    factor = 1
                    if abs(epoch_loss - best) > delta and epoch_loss < best:
                        best = epoch_loss
                        bad_epochs = 0
                        best_params = self.state_dict()
                    else:
                        bad_epochs += 1
                        if bad_epochs == max_bad_epochs + 1:
                            break
                else:
                    if abs(acc - best) > delta and acc > best:
                        best = acc
                        bad_epochs = 0
                        best_params = self.state_dict()
                    else:
                        bad_epochs += 1
                        if bad_epochs == max_bad_epochs + 1:
                            break

                pbar.set_description(f'BAD:{bad_epochs:0>2} LOSS:{epoch_loss:.2f} ACC:{acc*100:.1f} BEST:{best*factor:.1f}')

                #if accum_loss < 2e-5:
                #    break
            print("Loading best params...")
            self.load_state_dict(best_params)
            self.eval()
        
    def pred_dataframe(self, dataframe: pd.DataFrame, indexes: List[int], batch_size=256, _verbose=False):
        out = []
        if _verbose:
            deco = tqdm.tqdm
        else:
            deco = lambda x: x
        with torch.cuda.amp.autocast():
            for x, _ in deco(self.get_batches(dataframe, indexes, batch_size=batch_size)):
                out.extend(self.net(x).argmax(dim=-1).cpu().flatten().tolist())
        return np.array(out)
        
    def pred(self, inputs, batch_size=256, _verbose=False):
        out = []
        if _verbose:
            deco = tqdm.tqdm
        else:
            deco = lambda x: x
        for x, _ in deco(self.get_batches(inputs, batch_size=batch_size)):
            out.extend(self.net(x.float()).argmax(dim=-1).cpu().flatten().tolist())
        return np.array(out)
    
    def save(self, name):
        torch.save(self.state_dict(), f"{name}.pt")
        with open(f"{name}.json", "w") as f:
            json.dump(self.features, f)


# The following code would run for a single K of K-Fold
# 
# ```python
# model = NgramModel(input_dim=Xs.shape[1], output_dim=C)
# model.fit(Xs, YCs, batch_size=256, delta=.005, max_bad_epochs=5, epochs=400)
# model.net.eval()
# out = model.pred(X2s)
# 
# ConfusionMatrixDisplay.from_predictions(YC2s, out)
# print(classification_report(YC2s, out))
# df = pd.DataFrame(zip(YC2s, out), columns=["C", "pred"])
# print(df.plot.box(by="C"))
# for i in range(5):
#     df[f"RectPred{i}"] = abs(df["pred"] - df["C"]) <= i
#     counts = df[f"RectPred{i}"].value_counts()
#     print(f"Accuracy of CER predicted within {int(i*cer_of_bin*100)} " 
#           f"of the GT: {counts.get(True, 0)/(counts.get(False, 0)+counts.get(True, 0))*100:.2f}%")
#     
# df[f"RectPredReadable"] = (df["pred"] < 2) & (df["C"] < 2)
# counts = df[f"RectPred{i}"].value_counts()
# print(f"Accuracy of predicted CER < 10% (Readable) " 
#       f"of the GT: {counts.get(True, 0)/(counts.get(False, 0)+counts.get(True, 0))*100:.2f}%")
# ```

# In[10]:


import matplotlib.pyplot as plt
import random

def get_k_iterators(Ks, df):
    def ret():
        random.shuffle(Ks)
        for k in ks:
            ids, YCs, YNCs, Xs, XTranscriptions = get_features(train)
            yield Xs, YCs
    return ret

def make_for_K(K, df):
    train, dev, test = get_kfold_train_test(df, K)
    
    model = NgramModel(
        output_dim=len(df.bin.unique()),
        features=tuple([col for col in df.columns if col.startswith("$")])
    )
    print(model)
    model.fit(
        df,
        train_indexes=train,
        dev_indexes=dev,
        batch_size=1024,
        delta=.01,
        max_bad_epochs=10,
        epochs=2,
        lr=1e-2
    )
    model.net.eval()
    
    out = model.pred_dataframe(df, indexes=test[:2048], _verbose=True)
    test_truthes = df.loc[test[:2048], "bin"].tolist()
    
    # This should plot...
    figure, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ConfusionMatrixDisplay.from_predictions(test_truthes, out, ax=ax)
    plt.show()
    print(classification_report(test_truthes, out))
    e_df = pd.DataFrame(zip(test_truthes, out), columns=["bin", "pred"])
    print(e_df.plot.box(by="bin"))
    for i in range(5):
        e_df[f"RectPred{i}"] = abs(e_df["pred"] - e_df["bin"]) <= i
        counts = e_df[f"RectPred{i}"].value_counts()
        print(f"Accuracy of CER predicted within {int(i*5)} "
              f"of the GT: {counts.get(True, 0)/(counts.get(False, 0)+counts.get(True, 0))*100:.2f}%")

    e_df[f"RectPredReadable"] = (e_df["pred"] < 2) & (e_df["bin"] < 2)  # That's wrong ?
    counts = e_df[(e_df["bin"] < 2)]["RectPredReadable"].value_counts()
    print(f"Accuracy of predicted CER < 10% (Readable) "
          f"of the GT: {counts.get(True, 0)/(counts.get(False, 0)+counts.get(True, 0))*100:.2f}%")
    
    e_df[f"RectPred85"] = (e_df["pred"] < 3) & (e_df["bin"] < 3)  # That's wrong ?
    counts = e_df[(e_df["bin"] < 2)]["RectPred85"].value_counts()
    print(f"Accuracy of predicted CER < 15% (Readable) "
          f"of the GT: {counts.get(True, 0)/(counts.get(False, 0)+counts.get(True, 0))*100:.2f}%")
    
    model.save(f"ngram-{K}")
    del model


# In[11]:


NB_EXPS = 5
for i in range(KS):
    print(f"Dealing with {i}")
    make_for_K(K=i, df=df)
    if i+1 == NB_EXPS:
        break


# In[ ]:





# In[ ]:





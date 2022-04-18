"""
STUFF TO FILL



"""

"""
FOR GNB -- this stuff needs to be there 
will remove this comment later
"""

# Import in-built modules
import math
import random
import pandas as pd
import numpy as np

# Import User-Defined Modules
from GNB import GNB



"""
This stuff can be put in another file or something because it is tooooo UGLYYYY!!!!!
FROM HERE
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_df = pd.read_csv("./output/raw_counts.csv", index_col=0)
data_df

dict_map = {"Astrocytoma, NOS":"Astrocytoma", "Astrocytoma, anaplastic": "Astrocytoma",
            "Oligodendroglioma, NOS":"Oligodendroglioma","Oligodendroglioma, anaplastic": "Oligodendroglioma",
            "Glioblastoma":"Glioblastoma","Gliosarcoma":"Gliosarcoma"}

dropped_files = set()
label_df = pd.read_csv('./output/label.csv',index_col=0)
dropped_files.update(label_df[label_df["final_label"].isna()]["File Name"])
label_df = label_df[~label_df["final_label"].isna()].reset_index(drop = True)
dropped_files.update(label_df[label_df["final_label"] == 'Gliosarcoma']["File Name"])
label_df = label_df[label_df["final_label"] != 'Gliosarcoma'].reset_index(drop = True)
dropped_files.update(label_df[label_df["final_label"] == 'Mixed glioma']["File Name"])
label_df = label_df[label_df["final_label"] != 'Mixed glioma'].reset_index(drop = True)
label_df["labelf"] = label_df["final_label"].apply(lambda x: dict_map[x])
label_df

data_df = data_df[data_df.index.isin(label_df["File Name"])]
data_df

set(label_df['labelf'])

dict_map = {"Astrocytoma, NOS":"Astrocytoma", "Astrocytoma, anaplastic": "Astrocytoma",
            "Oligodendroglioma, NOS":"Oligodendroglioma","Oligodendroglioma, anaplastic": "Oligodendroglioma",
            "Glioblastoma":"Glioblastoma","Gliosarcoma":"Gliosarcoma"}

# # Some future Supplemental Data

# ### I have checked these replicates and these seem to have slightly different gene expression data, therefore it might be beneficial to have these as training data and test

rep = label_df.groupby('Case ID').agg(list).reset_index()
rep['len'] = rep.apply(lambda x: len(x['File Name']), axis=1)
rep = rep[rep['len']>1].reset_index(drop = True)
rep

data_df


# Using standard scaler to better PCA to 50 Components

X = data_df

sc = StandardScaler()
X = sc.fit_transform(X)

pca = PCA(n_components=280)
X = pca.fit_transform(X)
X.shape

df = pd.DataFrame()
df["y"] = label_df['labelf']
df["comp-1"] = X[:,0]
df["comp-2"] = X[:,1]

"""
TO HERE --> shift to another file --> TOO UGLY!!!!!
"""






"""
This is stuff I need in the main .py file
I need this Y thing and the gene expression data after feature selection as a numpy array
"""
# Get Y
Y_str = (label_df.labelf).to_numpy()
new_label = {'Glioblastoma':1, 'Astrocytoma':2, 'Oligodendroglioma':3}
Y = label_df["labelf"].apply(lambda x: new_label[x])
Y = Y.to_numpy()

"""
This is how I import GNB and implement it
"""
gnb = GNB(X, Y)
acc = gnb.ImplementGNB()







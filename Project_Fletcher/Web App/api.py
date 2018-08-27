import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

df_rec = pd.read_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df_rec.pkl')
df = pd.read_pickle('/Users/ajdavis/github/metis/Project_Fletcher/Data/df.pkl')

ss = StandardScaler()
transformed_data = ss.fit_transform(df_rec)
df_rec = pd.DataFrame(transformed_data, columns=df_rec.columns,
                      index = df_rec.index)

dists = cosine_similarity(df_rec)
dists = pd.DataFrame(dists, columns=df_rec.index)
dists.index = dists.columns



def get_similar(strains_dict, n=1):



    """
    calculates which strains are most similar to the input strains.
    Must not return the strains that were inputted.

    Parameters
    ----------
    strains: list
        some strains!

    Returns
    -------
    ranked_strains: list
        rank ordered strains
    """
    strains = [strains_dict['Strain1'], strains_dict['Strain2'], strains_dict['Strain3']]
    des_dict = pd.Series(df.Description.values,index=df.TestStrain).to_dict()
    strains = [strain for strain in strains if strain in dists.columns]
    strains_summed = dists[strains].apply(lambda row: np.sum(row), axis=1)
    strains_summed = strains_summed.sort_values(ascending=False)
    ranked_strains = strains_summed.index[strains_summed.index.isin(strains)==False]
    ranked_strains = ranked_strains.tolist()

    recs = ranked_strains[:n]
    descs = [des_dict[k] for k in recs]

    rec = recs + [':'] + [' '] + descs
    result = {
    'rec': rec
    }
    return result

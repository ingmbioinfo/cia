import numpy as np
import pandas as pd
import scanpy as sc

def celltypist_majority_vote(data, classification_obs, groups_obs=None, min_prop=0, unassigned_label='Unassigned'):
    """
    A function that wraps celltypist majority vote method. It extends the the more represented cell type label (predicted by a given method) to each reference cell groups.
    If reference cell groups are not provided it exploits scanpy.tl.leiden to over-clustering the dataset (this requires having ran scanpy.pp.neighbors or scanpy.external.pp.bbknn before).
    
    Parameters
    ----------
    
         
    data: anndata.AnnData
        an AnnData object.
        
    classification_obs: str or list(str)
        a string or a list of string specifying the AnnData.obs column/s where the labels assigned by the method/s of interest are stored.
        
    groups_obs: str or None
        a string specifying the AnnData.obs where the labels assigned by the reference method are stored. If None a over-clustering step with leiden algorithm is performed.     
        
    min_prop: float
        for the dominant cell type within a cell group, it represent e minimum proportion of cells required to support naming of the cell group by this cell type.
        (Default: 0, range: from 0 to 1)
    
    unassigned_label: str
        a string that specifies the label to assign to those cell groups in which none of the cell types reached the minimum proportion. 
       
    """

    if groups_obs==None:
        if data.n_obs < 5000:
            resolution = 5
        elif data.n_obs < 20000:
            resolution = 10
        elif data.n_obs < 40000:
            resolution = 15
        elif data.n_obs < 100000:
            resolution = 20
        elif data.n_obs < 200000:
            resolution = 25
        else:
            resolution = 30
        print('Reference annotation not selected.')
        print('Computing over-clustering with leiden algorithm (resolution= '+str(resolution)+') ...') 
        sc.tl.leiden(data, resolution=resolution, key_added='leiden_'+str(resolution))
        groups_obs='leiden_'+str(resolution)
        print('Dataset has been divided into '+str(len(data.obs[groups_obs].cat.categories))+' groups accordingly with trascriptional similarities.')
        print('')
        print('Over-clustering result saved in AnnData.obs["'+groups_obs+'"].')
    else: 
        print('AnnData.obs["'+groups_obs+'"] selected as reference annotation.')
    
    print('Extending the more represented cell type label to each cell group...')
    print('')    
    groups=np.array(data.obs[groups_obs])
    
    if type(classification_obs)!=list:
        classification_obs= list(classification_obs)
    for i in classification_obs:       
        votes = pd.crosstab(data.obs[i], groups)
        majority = votes.idxmax(axis=0)
        freqs = (votes / votes.sum(axis=0).values).max(axis=0)
        majority[freqs < min_prop] = 'Unassigned'
        majority = majority[groups].reset_index()
        majority.index =data.obs[groups_obs].index
        majority.columns = [groups_obs, 'majority_voting']
        majority['majority_voting'] = majority['majority_voting'].astype('category')
        data.obs[i+' majority voting']=majority['majority_voting']
        print('New classification labels have been stored in AnnData.obs["'+i+' majority voting"]. ')
        print('')
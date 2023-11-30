import numpy as np
import pandas as pd
from anndata import AnnData
import time 
import multiprocessing
import functools
from functools import partial
from scipy.sparse import issparse
from scipy import sparse
from itertools import islice
import random

def signature_score(signatures_dict, data, score_mode='raw', return_array=False):  

    """ 
    Given a colleEction of gene signatures, this function, for each signature, computes a score for each cell, which increases both when the number of expressed genes in the signature increases, and when the expression of these genes increases.
    

    Parameters
    ----------
    
    signatures_dict: dict or str
        a dictionary having as keys the signature names and as values the gene signatures (lists of gene names). In alternative, a filepath or a url of a tab separated gmt file without header can be provided too.
   
    data: anndata.AnnData
        an AnnData object.
        
    score_mode: str
        a string that specifies the type of scores to be computed. The accepted values are: "raw", "scaled", "log", "log2" and "log10".
        The "scaled" score is the raw score divided by the max value (scaled from 0 to 1). The other ones are the logarithm of the raw score with diffent base (e for log, 2 for log2 and 10 for log10).
        
    return_array: bool
        a boolean that specifies if the scores have to be returned as a numpy.ndarray (True) or if each signature score must be added to data.obs (False).
        
    
     
    Returns
    -------
    
    scores: numpy.ndarray
        a numpy.ndarray containing signature scores. NB: scores is returned only if return_array is True.
    
    
    Raises
    ------
    
    ValueError
        If score_mode is not one of these values:  ['raw', 'scaled', 'log', 'log2', 'log10'].
        
    TypeError
        If signatures_dict is not a dict.
        If return_array is not a boolean. 

    """

    index=[]
    scores=[]

    # check input data
    if score_mode not in ['raw', 'scaled', 'log', 'log2', 'log10']:
        raise ValueError("score_mode must be one of these values: ['raw', 'scaled', 'log', 'log2', 'log10'] ")

    if type(signatures_dict) not in [dict, np.ndarray, str]:
        raise TypeError("signatures_dict must be a dict or a path to a gmt with no header!")

    if ((type(signatures_dict)!=dict)&(type(signatures_dict)!=str)):
        signatures_dict=dict(enumerate(signatures_dict, 0))

    # return data
    if type(return_array)!=bool:
        raise TypeError("return_array must be a boolean")
    if return_array==False:   
        print('Checking for genes not in AnnData.raw.var_names ...')
        print('')
    c=0 #counter

    if type(signatures_dict)==str:
        df=pd.read_csv(signatures_dict, sep='\t', header=None)
        signatures_dict={}
        if sum([len(i.split())>1 for i in df.iloc[:,1]])>0:
            for i in range(0, len(df.index)):
                signatures_dict[df.iloc[i,0]]=list(df.iloc[i,2:].dropna())         
        else:
            for i in range(0, len(df.index)):
                signatures_dict[df.iloc[i,0]]=list(df.iloc[i,1:].dropna())
        
    # Actually compute the score for each signature
    for i in signatures_dict.keys():
        index.append(i)
        geneset = signatures_dict[i]
        geneset_original_type = type(geneset) #allow to visualize excluded genes number when using user signatures
        if type(geneset) != np.ndarray:
            geneset = np.array(geneset)
        # Filter out None values from geneset and not present in anndata
        geneset = geneset[geneset != None]
        geneset=np.intersect1d(geneset, np.array(data.raw.var_names)) #NOTE:genes' order is not mantained
        #return(geneset, signatures_dict[i])
        if (len(geneset) != len(signatures_dict[i])) & (geneset_original_type != np.ndarray):
            c+=1
            print(str(len(signatures_dict[i]) - len(geneset)) + '/' + str(len(signatures_dict[i])) + ' of "' + str(i) + '" signature genes were removed since they are not in AnnData.raw.var_names')
        # matrix of count (true when the expression of a gene in the geneset is higher than 0)
        count = data.raw[:, geneset].X > 0
        # matrix of raw expresion values of genes of the geneset, row number of genes in adata and col number if genes in the signature
        exp = data.raw[:, geneset].X 
        count = count.sum(axis=1)/len(geneset) # vector of count: row-wise (per cell) sum divided by the total number of genes in geneset
        exp = exp.sum(axis=1)/data.raw[:, :].X.sum(axis=1)
        # vector of expression: rowe-wise (per cell) sum divided by the total expression of all the genes.
        score = np.array(count)*np.array(exp) # to compute signature score
        score = score.reshape(score.shape[0],1)
        scores.append(score)
 
    if return_array == False:   
        if c == 0:
            print('All signature genes are in AnnData.raw.var_names')
    
    scores = pd.DataFrame(np.hstack(scores), columns=index)

    if score_mode =='scaled':
        scores = scores/ scores.max()
        
    elif score_mode == 'log2':
        min_val = np.nextafter(np.float32(0), np.float32(1))
        scores = np.log2(scores + min_val)
        
    elif score_mode == 'log10':
        min_val = np.nextafter(np.float32(0), np.float32(1))
        scores = np.log10(scores + min_val)
    
    elif score_mode =='log':
        min_val = np.nextafter(np.float32(0), np.float32(1))
        scores = np.log(scores + min_val)
        
    
    if return_array == True: 
        scores = scores.to_numpy()
        return scores
    else:
        print('')
        print('Computing ' + score_mode + ' signature scores ...')
        print('')
        for i in scores.columns:
            data.obs[i] = list(scores[i])
            print('"' + str(i) + '"' + ' added in Anndata.obs')


def _shuffled(array: list, seed: int = None) -> list:
    
    array2 = array.copy()
    if seed is None:
        seed = random.randint(1, 0xFFFE)

    length_ = len(array)
    shuffled = [0]*length_
    seed_pos = seed ^ 0xFFFF

    for i in list(reversed(range(length_))):
        index = seed_pos % (i + 1)
        shuffled[i] = array2.pop(index)

    return shuffled

def _flatter(ddl):
    # Determine the maximum length of inner lists in all dictionaries
    max_inner_list_length = max(len(inner_list) for d in ddl for inner_list in d.values())

    # Initialize an empty list to store data in the desired order
    result_list = []

    # Iterate through the dictionaries and store data in the desired order
    for d in ddl:
        flattened_data = []
        for key in sorted(d.keys()):  # Sort the keys for consistent order
            inner_list = d[key]
            flattened_data.extend(inner_list)
            # Pad with None to make all inner lists the same length
            flattened_data.extend([None] * (max_inner_list_length - len(inner_list)))
        result_list.append(flattened_data)

    # Calculate the total number of elements in the flattened data
    total_elements = sum(len(row) for row in result_list)

    # Initialize a 2D NumPy array filled with None
    result_array = np.full((len(result_list), total_elements), None, dtype='object')

    # Fill the 2D array with values from the lists in result_list
    for i, row in enumerate(result_list):
        result_array[i, :len(row)] = row

    return result_array

def _stack(ddl):

    # Determine the number of lists (e.g., 'A', 'B', 'C', etc.)
    num_lists = len(ddl[0])
    
    # Determine the maximum length of inner lists in all dictionaries
    max_inner_list_length = max(len(inner_list) for d in ddl for inner_list in d.values())

    # Initialize a list to store data in the desired order
    result_list = []

    # Iterate through the dictionaries and store data in the desired order
    for d in ddl:
        for key in d.keys():  # Sort the keys for consistent order
            # Pad with None to make inner lists the same length
            inner_list = d[key]
            inner_list.extend([None] * (max_inner_list_length - len(inner_list)))
            result_list.append(inner_list)
    
    # Convert the list of lists into a 2D NumPy array
    result_array = np.array(result_list, dtype='object')

    return result_array


def _parallel_apply_along_axis(func1d, axis, arr, n_proc, *args, **kwargs):

    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)
    chunks = [sub_arr for sub_arr in np.array_split(arr, n_proc) if sub_arr.shape[0]>0]
    print("Number of chunks:", len(chunks))

    pool = multiprocessing.Pool(processes=n_proc)
    prod_x = partial(func1d, data=kwargs['data'], return_array=kwargs['return_array'])
    individual_results = pool.map(prod_x, chunks)
    pool.close() 
    pool.join()

    #return individual_results #add to test
    new_arr = np.concatenate(individual_results, axis=1)

    return new_arr


def generate_random_gmt(signature, data, obs_cut_df):

    #signature = np.intersect1d(np.array(signatures), np.array(data.raw.var_names))

    gene_pool = list(data.var_names[~ data.var_names.isin(signatures)])
    random_scores = []
    X_list = data.obs[i].to_numpy()
    X_list = X_list.reshape(X_list.shape[0],1)
    random_gmt = {}

    ### slicing obs_cut to it ot obatin genelist 2 bin and also to know quanti geni random 
    #devo prendere per ogni bin
    n_rand = obs_cut_df.loc[signature,:].groupby('bin').size()

    gene2retain = obs_cut_df.loc[gene_pool,:]

    #dictionrary bin-genes
    obs_dict={}
    for i in gene2retain['bin'].unique():
        obs_dict[i]=gene2retain[gene2retain['bin']==i].index.tolist()
    for i in (set(obs_cut_df['bin'].unique()) - set(gene2retain['bin'].unique())):
        obs_dict[i]=obs_cut_df[obs_cut_df['bin']==i].index.tolist()

    for j in range(0,n_iter):
        random_list=[]
        for bins in n_rand.index:
            obs_dict[bins] = _shuffled(obs_dict[bins], seed)
            random_list=random_list+ obs_dict[bins][:n_rand[bins]]
            seed = seed + 1
        random_gmt[str(j)]=random_list

    return random_gmt


def signature_based_classification(data, signatures_dict=dict(), negative_markers=None, fast_mode=False, partial_classification=False, n_iter=500 ,p=None, q=None, FC_threshold=1, new_score=None,obs_name='Class', n_proc=4, n_bins=25, unassigned_label='Unassigned', seed=42):
    
    """ 
    Given a dictionary of gene signatures, this function exploits signature_score() to compute scaled score (with fast_mode=True) or Fold Change
    (FC) scores of each signature (signature score/randomic signature score) and it classifies each cell accordingly with the highest score
    value. 
 
          
    Parameters
    ----------
    
    data: anndata.AnnData
        an AnnData object.
    
    signatures_dict: dict or str
        a dictionary having as keys the signature names and as values the gene signatures (lists of gene names). In alternative, a filepath or a url of a tab separated gmt file without header can be provided too.
        
    negative_markers: dict or None
        a dictionary having as keys the signature names and as values the lists of negative marker genes (lists of gene names).

    fast_mode: bool
        if True the comparison with randomic genes is skipped.
        
    partial_classification: bool
        Only when fast_mode is True it allows to specify if the signatures are expected to classify the whole dataset (False) or not (True).

    n_iter: int
        an integer to set the number of iterations performed to generate the random gene lists.
        
    p: int or float or None
        if not None, p is a number that specifies the p-value above which the signature is not considered significant in the cell.
        
    q: int or float or None
        if not None, q is a number (from 0 to 1) that specifies the quantile of a signature FC score values distribution, below which the signature is not considered expressed in the cell.
        
    FC_threshold: int or float
        a number that specifies the minimum FC score value to be considered.
        NB: the default value is 1, meaning that when the FC score is lower than 1 (the signature score is lower than the randomic one), the FC score value is set to 0. 
    
    new_score: str or None
        if not None, new_score specifies if raw scores in obs have to be substituted with FC score ("FC"), significance threshold ("thr") or filtered FC scores ("filtered").
    
    obs_name: str 
        a string that specifies the AnnData obs name of the returned classification.
        
    n_proc: int
        an integer that specifies the number of processors to be used.
        
    n_bins: int
        an integer that specifies the number of expression level bins in which dataset genes are divided.
       
    unassigned_label: str
       a string that specifies the label of unissigned cells.
    seed: int or None
        A numeric indicating a seed for creating randomic signatures aimed at reproducible analyses
       
    
    Raises
    ------
    
    TypeError
        If n_iter is not an integer.
        If p is not an integer or a float.
        If q is not an integer or a float.
        If FC_threshold is not an integer or a float.
        If n_bins is not an integer.
        
    ValueError
        If q is not between 0 and 1.
    
    KeyError
        If a negative_markers key doesn't match signatures_dict keys. 
    
     
    """
    if type(signatures_dict)==str:
        df=pd.read_csv(signatures_dict, sep='\t', header=None)
        signatures_dict={}
        if sum([len(i.split())>1 for i in df.iloc[:,1]])>0:
            for i in range(0, len(df.index)):
                signatures_dict[df.iloc[i,0]]=list(df.iloc[i,2:].dropna())         
        else:
            for i in range(0, len(df.index)):
                signatures_dict[df.iloc[i,0]]=list(df.iloc[i,1:].dropna())
                    
    signatures = signatures_dict.copy()
    start = time.time()

    if negative_markers!=None:
        if type(negative_markers) !=dict:
            raise TypeError("negative_markers must be a dict")
        if len(np.intersect1d(list(negative_markers.keys()),list(signatures.keys())))!=len(list(negative_markers.keys())):
            raise KeyError("negative_markers.keys() must be also signatures_dict.keys()")

    if fast_mode==True:
        if partial_classification==True or len(signatures.keys())==1:
            ctrl=data.var_names
            for i in signatures.keys():
                ctrl=list(set(ctrl)-set(signatures[i]))
            signatures[unassigned_label]= ctrl
        
        arr=signature_score(data=data, signatures_dict=signatures, score_mode='scaled',  return_array=True)
        data.obs[list(signatures.keys())]=arr

        if negative_markers!=None:
            nm={}
            for n in negative_markers.keys():
                nm[n+'_negative']=negative_markers[n]
            signature_score(data=data, signatures_dict=nm, score_mode='scaled')
            for n in negative_markers.keys():
                kl=np.array(list(signatures.keys()))
                arr[:,np.where(kl==n)[0][0]]=arr[:,np.where(kl==n)[0][0]]*np.array(1-data.obs[n+'_negative'])      
        pred=np.argmax(arr, axis=1)
        pred_str=pred.astype('O')
        for n in range(0, arr.shape[1]):
            pred_str[pred==n]=list(signatures.keys())[n]

        pred_str[np.max(arr, axis=1)==0] = unassigned_label
        print('')
        print('Classification labels added in AnnData.obs["' + obs_name + '"]')

        data.obs[obs_name]=pred_str
        end = time.time()
        print('')
        print(f"Runtime of the process is {round((end - start),2)} s ")
        return "Fast classification complete!"

    # Check inputs   
    if type(n_iter) != int:
        raise TypeError("n_iter must be an integer")
    if p != None:
        if type(p) not in [int, float]   :
            raise TypeError("p must be an integer or a float")
    if q != None:        
        if type(q) not in [int, float]   :
            raise TypeError("q must be an integer or a float")
        if q < 0 or q > 1 :
            raise ValueError("q must be between 0 and 1")
        
    if type(FC_threshold) not in [int, float]   :
        raise TypeError("FC_threshold must be an integer or a float")
        
    if type(n_bins) != int:
        raise TypeError("n_bins must be an integer")
    
    # Compute signature score and store the results in the adata
    signature_score(data=data, signatures_dict=signatures)

    # Generation of the mean of the gene dataset
    if not issparse(data.raw.X):
        matrix = sparse.csr_matrix(data.raw.X)
        print("")
        print('convert dense to sparse matrix')
    else:
        matrix = data.raw.X
            
    data.raw.var['bin']=np.mean(matrix, axis=0).tolist()[0]
    
    obs_avg=data.raw.var['bin']
    
    # Division to obtain the genes in the bin
    n_items = int(np.round(len(obs_avg) / n_bins))
    obs_cut = obs_avg.rank(method='min') // n_items
    obs_cut_df = pd.DataFrame(obs_cut)

    # Define list which contains partial results from randomization for each signature
    min_val = np.nextafter(np.float32(0), np.float32(1))
    index = []
    pval_matrix = []
    FC_scores = []
    st = []
    
    # Generate random gmt for each signatures
    random_gmt_list = [] 
    for i in signatures.keys():
        
        #geneset=signatures_dict[i]
        signatures[i]=np.intersect1d(np.array(signatures[i]), np.array(data.raw.var_names))
    
        index.append(i)
        gene_pool = list(data.var_names[~ data.var_names.isin(signatures[i])])#list(set(data.var_names) - set(signatures[i]))
        random_scores = []
        X_list = data.obs[i].to_numpy()
        X_list = X_list.reshape(X_list.shape[0],1)
        random_gmt = {}
         
        ### slicing obs_cut to it ot obatin genelist 2 bin and also to know quanti geni random 
        #devo prendere per ogni bin
        n_rand = obs_cut_df.loc[signatures[i],:].groupby('bin').size()
    
        gene2retain = obs_cut_df.loc[gene_pool,:]

        #dictionrary bin-genes
        obs_dict = {}
        for i in gene2retain['bin'].unique():
            obs_dict[i] = gene2retain[gene2retain['bin']==i].index.tolist()
        
        for i in (set(obs_cut_df['bin'].unique()) - set(gene2retain['bin'].unique())):
            obs_dict[i] = obs_cut_df[obs_cut_df['bin']==i].index.tolist()

        for j in range(0,n_iter):
            random_list = []
            for bins in n_rand.index:
                obs_dict[bins] = _shuffled(obs_dict[bins], seed)
                random_list = random_list + obs_dict[bins][:n_rand[bins]]
                seed = seed + 1
            random_gmt[str(j)]=random_list
        random_gmt_list.append(random_gmt)

    # Convert the dictionary of dictionary of list to a matrix stacking random gene signatures along rows,
    # and columns equal to the number of genes in the signature 
    random_gmt_mtx = _stack(random_gmt_list)

    # Compute signature score for each randomic signature
    random_scores_mtx = _parallel_apply_along_axis(signature_score, axis=1, arr=random_gmt_mtx, 
        n_proc=n_proc, data=data, return_array=True)
    
    #need to split in submatrix corresponding to the initial sublist
    # Number of subarrays (number of splits)
    n_sub = len(signatures.keys())
    # Split the array into n subarrays along the columns
    subarrays = np.array_split(random_scores_mtx, n_sub, axis=1)
    
    #return subarrays
    
    for j in range(0, len(subarrays)):
        # compute median among each iteration (random signatures) obtaining a 2d array with 1 column and rows equal to the cell number
        X_control = np.median(subarrays[j], axis=1)
        X_control = X_control.reshape(X_control.shape[0],1)
        
        #get true signature vlues from adata
        X_list = data.obs[list(signatures.keys())[j]].to_numpy()
        X_list = X_list.reshape(X_list.shape[0],1)
        
        if p != None:
            rt = np.quantile(subarrays[j], 1-p, axis=1)
            st.append(rt)
            pval_matrix.append((X_list>rt)[:,0])
            
        X_list = X_list + min_val
        #return X_list
        X_control = X_control + min_val
        score = X_list/X_control
        FC_scores.append(score)

    pval_matrix = np.array(pval_matrix)
    st = np.array(st)
    FC_scores = np.array(FC_scores)
    FC = FC_scores.copy()
    uns1 = pd.DataFrame(FC.squeeze())
    uns2 = pd.DataFrame(np.array([[FC_threshold]*len(data.obs.index)]*len(signatures.keys())).squeeze())

    if q != None:
        uns2=[]
        for l in range(0, len(index)):
            FC_scores[l][(FC_scores[l]) < np.quantile(FC_scores[l], q=q)]=0
            uns2.append([np.quantile(FC_scores[l], q=q)]*len(data.obs.index))
        uns2= np.array(uns2)
        uns2= pd.DataFrame(uns2.squeeze())
    
    if p != None:
        uns2=[]
        for l in range(0, len(index)):
            FC_scores[l][np.logical_not(pval_matrix[l])] = 0
        uns2= pd.DataFrame(st.squeeze())

    for l in range(0, len(index)):
        FC_scores[l][(FC_scores[l]) <= FC_threshold] = 0

    if negative_markers!=None:
        nm={}
        for n in negative_markers.keys():
            nm[n+'_negative']=negative_markers[n]
        signature_score(data=data, signatures_dict=nm, score_mode='scaled')
        kl=np.array(list(signatures.keys()))
        for n in negative_markers.keys():
            FC_scores[np.where(kl==n)[0][0]]=np.diagonal(FC_scores[np.where(kl==n)[0][0]]*np.array(1-data.obs[n+'_negative'])).reshape(len(data.obs),1)
    
    # choose the best annotation
    pred = np.argmax(FC_scores, axis=0)
    pred_str = pred.copy()
    pred_str = pred_str.astype('O')
    
    uns3 = pd.DataFrame(FC_scores.squeeze())
    
    #update prediction with corresponding label
    for n in range(0, len(index)):
         pred_str[pred==n] = index[n]
    
    pred_str[np.max(FC_scores, axis=0)==0] = unassigned_label
    
    print('')
    print('Classification labels added in AnnData.obs["' + obs_name +'"]')       
    data.obs[obs_name] = pred_str
    
    if len(signatures.keys())!=1:
        
        uns1.index=[i+'_FC' for i in signatures.keys()]
        uns2.index=[i+'_thr' for i in signatures.keys()]
        uns3.index=[i+'_filtered' for i in signatures.keys()]
    
        print('')
        print('Results have been stored in AnnData.uns["signature_based_classification"]')
        data.uns['signature_based_classification']=pd.concat([uns1, uns2, uns3]).transpose()
    
    if len(signatures.keys())==1:
            
        print('')
        print('Results have been stored in AnnData.uns["signature_based_classification"]')
        data.uns['signature_based_classification']=pd.concat([uns1, uns2, uns3], axis=1)
        data.uns['signature_based_classification'].columns=[list(signatures.keys())[0] + '_FC',
                                                             list(signatures.keys())[0] + '_thr',
                                                             list(signatures.keys())[0] + '_filtered']

    if new_score != None:
        
        if new_score not in ['FC','thr', 'filtered']:
            raise ValueError('new_score must be "FC","thr" or "filtered"')
            
        elif new_score == 'FC':
            print('')
            print('raw scores are being replaced by Fold Change signature scores ...')
            print('')
            data.obs[list(signatures.keys())]=data.uns['signature_based_classification'][[i+'_FC' for i in signatures.keys()]].values

        elif new_score == 'thr':
            print('')
            print('raw scores are being replaced by significance treshold ...')
            print('')
            data.obs[list(signatures.keys())]=data.uns['signature_based_classification'][[i+'_thr' for i in signatures.keys()]].values

        elif new_score == 'filtered':
            print('')
            print('raw scores are being replaced by filtered Fold Change signature scores ...')
            print('')
            data.obs[list(signatures.keys())]=data.uns['signature_based_classification'][[i+'_filtered' for i in signatures.keys()]].values

    # end time
    end= time.time()
    print('')
    print(f"Runtime of the process is {round((end - start)/60,2)} min with {n_proc} cores")

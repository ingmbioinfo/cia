import numpy as np
import pandas as pd
import itertools 
import scipy

def signatures_similarity(signatures_dict, show='J'):
    
    """ 
    A function that computes the similary between gene signatures.
           
    Parameters
    ----------
     
    
    signatures_dict: dict
        a dictionary having as keys the signature names and as values the gene signatures (lists of gene names).
        
    show: str
        a string that specifies if similarities will be returned as Jaccard index ("J") or as percentages of intersection ("%").
    
    Returns
    -------
    
    similarity: pandas.DataFrame
        a pandas.DataFame containing the similarity of each pair of signatures.
        
    Raises
    ------
    
    ValueError
        if show is different from "J" or "%".
    """


    similarity={}
    for i in signatures_dict.keys():
        values=[]
        if show=='J':
            for j in signatures_dict.keys():
                values.append(len(np.intersect1d(signatures_dict[i], signatures_dict[j]))/len(np.union1d(signatures_dict[i], signatures_dict[j])))
        elif show=='%':
            for j in signatures_dict.keys():
                values.append(round(100*len(np.intersect1d(signatures_dict[i], signatures_dict[j]))/len(signatures_dict[i]),2))
            
        else:
            raise ValueError('show must be "J" or "%".')
        similarity[i]=values
    similarity=pd.DataFrame(similarity)
    similarity.index= similarity.columns
    return similarity


def filter_degs(data, groupby, uns_key='rank_genes_groups', direction='up', logFC=0,  scores=None, perc= 0, mean=0):
    
    """ 
    A function that filters differentially expressed genes (DEGs) obtaied with scanpy.tl.rank_genes_groups.
           
    Parameters
    ----------

    data: anndata.AnnData
        an AnnData object.     
    
    groupby: str
        a string that specifies the AnnData.obs column containing cell groups of the differential expression analysis.

    uns_key: str
        a string that specifies the AnnData.uns where differential expression analysis results are stored.
    
    direction: str
        a string that specifies if filtering-in above thersholds ("up", to select upregulated genes) or below thresholds ("down", to select down regulated genes).
    
    logFC: int or float
        Log fold-change threshold.

    scores: int, float or None
        Z score threshold.
    
    perc: int or float
        Threshold of the percentage of expressing cell within the cell group.
    
    mean: int or float
        Mean expression threshold.
        
    
    Returns
    -------
         
    signatures_dict: dict
        a dictionary having as keys the cell group names and as values the filtered-in DEGs  (lists of gene names).
        
    Raises
    ------
    
    ValueError
        if direction is different from "up" or "down".
    """

    signatures_dict={}
    for group in data.obs[groupby].cat.categories:
    #for group in data.uns[uns_key]['names'].dtype.names:
        degs=data.uns[uns_key]['names'][group] 


        n_cells=sum(data.obs[groupby]==group)


        
        if direction=='up':
            
            order=pd.DataFrame(data.uns[uns_key]['logfoldchanges'][group]).sort_values(by=0,ascending=False).index
            degs= degs[order]

            if scipy.sparse.issparse(data.raw.X):
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.todense() > 0).sum(axis=0)/n_cells*100)
            else:
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X > 0).sum(axis=0)/n_cells*100)
            cells =(cells >= perc)
            
            gene_mean = np.ravel(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.mean(0))
            gene_mean = (gene_mean >= mean )
            
            lfc= data.uns[uns_key]['logfoldchanges'][group]
            lfc= (lfc[order] >=logFC)
            
            filters=[cells, gene_mean, lfc]
            
            if scores!=None:
                s= data.uns[uns_key]['scores'][group]
                s= (s[order] >=scores)
                filters.append(s)
            

            filters=np.bitwise_and.reduce(filters)
            signatures_dict[group]= degs[filters].tolist()
            
            

            
        elif direction=='down':

            order=pd.DataFrame(data.uns[uns_key]['logfoldchanges'][group]).sort_values(by=0,ascending=False).index
            degs= degs[order]
            
            if scipy.sparse.issparse(data.raw.X):
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.todense() > 0).sum(axis=0)/n_cells*100)
            else:
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X> 0).sum(axis=0)/n_cells*100)
            cells =(cells <= perc)
            
            gene_mean = np.ravel(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.mean(0))
            gene_mean = (gene_mean <= mean )
            
            lfc= data.uns[uns_key]['logfoldchanges'][group]
            lfc= (lfc[order] <=logFC)
            
            filters=[cells, gene_mean, lfc]
            
            if scores!=None:
                s= data.uns[uns_key]['scores'][group]
                s= (s[order] <=scores)
                filters.append(s)
            

            filters=np.bitwise_and.reduce(filters)
            signatures_dict[group]= degs[filters].tolist()
            
        else:
            raise ValueError('direction must be "up" or "down".')
        
    return signatures_dict

def read_gmt(file, sep='\t', header=None):
    """
    A function to convert a gmt file (with or without description of signatures) into a dictionary.
    
    Parameters
    ----------
     
    file: str
        filepath of gmt file. See pandas.read_csv documentation.   
    sep: str
        delimiter to use. See pandas.read_csv documentation.
    header: int, list(ind) or None
        row number(s) to use as the column names, and the start of the data. See pandas.read_csv documentation. 

    Returns
    -------
    
    gmt: dict
        a dictionary having the signature names as keys and gene lists as values.
    """
    gmt={}
    df=pd.read_csv(file, sep=sep, header=header)
    if sum([len(i.split())>1 for i in df.iloc[:,1]])>0:
        for i in range(0, len(df.index)):
            gmt[df.iloc[i,0]]=list(df.iloc[i,2:].dropna())
            
    else:
        for i in range(0, len(df.index)):
            gmt[df.iloc[i,0]]=list(df.iloc[i,1:].dropna())
    return gmt

def save_gmt(signatures_dict, file):
    """
    A function to convert a dictionary of signatures in a gmt file correctly formatted for signature_score and signature_based_classification functions.
    
    Parameters
    ----------
     
    signatures_dict: dict
        a dictionary having as keys the signature names and as values the gene signatures (lists of gene names).
    file: str
        filepath of gmt file. See pandas.DataFrame.to_csv documentation.  
    """
    pd.DataFrame.from_dict(signatures_dict, orient='index').to_csv(file, sep='\t', header=None)

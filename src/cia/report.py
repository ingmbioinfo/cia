import numpy as np
import pandas as pd
import seaborn as sns
import itertools 
import scipy
import os

def group_composition(data, classification_obs, groups_obs, columns_order=None, cmap='Reds', save=None):
    
    """ 
    Plots a heatmap showing the percentages of cells classified with a given method (method of interest) in cell groups defined with a different one (reference method).
           
    Parameters
    ----------
     
    data: anndata.AnnData
        an AnnData object.
        
    classification_obs: str
        a string specifying the AnnData.obs where the labels assigned by the method of interest are stored.
    groups_obs: str
        a string specifying the AnnData.obs where the labels assigned by the reference method are stored.  
    columns_order: list(str)
        a list of strings with column names    
    cmap: str or matplotlib.colors.Colormap
        the mapping from data values to color space. 
    save: str
        a string specifying the file name. In the working directory, if not present, 'figures' directory will be created and a file with the prefix 'CIA_' will be saved in it.
    
    Returns
    -------
    
    sns.heatmap(): AxesSubplot
        a Axesubplot containing the percentages of cells classified with a given method in cell groups.
    """
    df=pd.crosstab(data.obs[groups_obs], data.obs[classification_obs])
    df=round((df/np.array(df.sum(axis=1)).reshape(len(df.index),1))*100,2)
    if columns_order!=None:
        df=df[columns_order]
        if save!=None:
            if not os.path.exists('./figures'):
                os.makedirs('figures')
            return sns.heatmap(df, cmap=cmap, annot=True).get_figure().savefig("figures/CIA_"+save)
    return sns.heatmap(df, cmap=cmap, annot=True)    
 

def grouped_distributions(data, columns_obs, groups_obs, cmap='Reds', scale_medians=None, save=None):
    
    """ 
    By selecting AnnData.obs columns, this function plots a heatmap showing the medians of their values in cell groups and it prints a statistical report. For each cell group, a two-sided Wilcoxon test is perfomed to evaluate if the distribution with the highest median is different from the others. For each selected AnnData.obs columns set of values, grouped by cell groups, a two-sided Mann-Whitney U test is performed to evaluate if the distribution in the cell group having the highest median is different from the other groups distributions.
           
    Parameters
    ----------
     
    data: anndata.AnnData
        an AnnData object.
    columns_obs: list(str)
        a string specifying the AnnData.obs columns where the values of interest are stored.
    groups_obs: str
        a string specifying the AnnData.obs where the cell labels are stored.    
    cmap: str or matplotlib.colors.Colormap
        the mapping from data values to color space. 
    scale_medians: str or None
        a parameter to set the scaling type of median values (None, 'row-wise', 'column-wise') 
    save: str
        a string specifying the file name. In the working directory, if not present, 'figures' directory will be created and a file with the prefix 'CIA_' will be saved in it.
    
    Returns
    -------
    
    sns.heatmap(): AxesSubplot
        a Axesubplot containing a heatmap of the score medians in cell groups.
    """
    grouped_df=data.obs.groupby(groups_obs).median()
    grouped_df=grouped_df[columns_obs]
    if scale_medians!=None:
        if scale_medians=='row-wise':
            grouped_df=grouped_df.transpose()/np.array(grouped_df.sum(axis=1))
            grouped_df=grouped_df.transpose()
        if scale_medians=='column-wise':
            grouped_df=grouped_df/np.array(grouped_df.sum(axis=0))
   
    
    subsets={}
    results={}
    print('Performing Wilcoxon test on each cell group ...')
    combs=list(itertools.permutations(columns_obs,2))
    count=0
    for i in data.obs[groups_obs].cat.categories:
        subsets[i]= data[data.obs[groups_obs]==i].obs[columns_obs]
        pos=subsets[i].median().values.argmax()
        
        for j in combs:
            if ((sum(subsets[i][j[0]])!=0) & (sum(subsets[i][j[1]])!=0)):
                result=scipy.stats.wilcoxon(subsets[i][j[0]], subsets[i][j[1]], alternative='two-sided')
                if result[1] >= 0.01 and j[0]==subsets[i].median().index[pos]:
                            count+=1
                            print('WARNING in cell group '+i+': '+ j[0]+' values are not significantly different from '+j[1]+' values.')
    if count==0:
        print('For each cell group there is a distribution significantly higher than the others (p<0.01)')

    print('')
    print('Performing Mann-Whitney U test on each selected AnnData.obs column ...')
    combs=list(itertools.permutations(data.obs[groups_obs].cat.categories,2))
    count=0
    for i in columns_obs:
        sign={}
        l=[]
        for c in data.obs[groups_obs].cat.categories:
            l.append(subsets[c][i].values)
        sign[i]=pd.DataFrame(l).transpose()
        sign[i].columns=data.obs[groups_obs].cat.categories
        pos=sign[i].median().argmax()
        for j in combs:
            result=scipy.stats.mannwhitneyu(subsets[j[0]][i], subsets[j[1]][i], alternative='two-sided')
            if result[1] >= 0.01 and j[0]==sign[i].median().index[pos]:
                count+=1
                print('WARNING in '+i+' distribution: values in '+ j[0]+' group are not significantly different from values in '+j[1]+' group')  
                print('(p= '+str(result[1])+')')
    if count==0:
        print('For each distribution, there is only a cell group in which values are higher with respect to all the other groups  (p<0.01)')
            

    if save!=None:
        if not os.path.exists('./figures'):
            os.makedirs('figures')
        return sns.heatmap(grouped_df, cmap=cmap, annot=True).get_figure().savefig("figures/CIA_"+save)
    return sns.heatmap(grouped_df, cmap=cmap, annot=True)    

def classification_metrics(data, classification_obs, groups_obs):
    """ 
    Computes the main metrics of classification by comparing labels of cells classified with given methods (methods of interest) and labels assigned with a different one (reference method).
    NB: labels must be correspondent!
           
    Parameters
    ----------
     
    data: anndata.AnnData
        an AnnData object.
        
    classification_obs: list(str)
        a list of string specifying the AnnData.obs columns where the labels assigned by the methods of interest are stored.
    groups_obs: str
        a string specifying the AnnData.obs where the labels assigned by the reference method.      
    
    Returns
    -------
    
    report: pandas.DataFrame
        a pandas.DataFame containing the overall sensitivity (SE), specificity (SP), precision (PR), accuracy (ACC) and F1-score (F1) of the selected classification methods.
    """
        
    report={}
    for m in classification_obs:
        SE=[]
        SP=[]
        PR=[]
        ACC=[]
        F1=[]



        TP_l=[]
        TN_l=[]
        FP_l=[]
        FN_l=[]

        for i in data.obs[groups_obs].cat.categories:
            TP_l.append(sum((data.obs[m]==i)& (data.obs[groups_obs]==i)))
            TN_l.append(sum((data.obs[m]!=i)& (data.obs[groups_obs]!=i)))
            FP_l.append(sum((data.obs[m]==i)& (data.obs[groups_obs]!=i)))
            FN_l.append(sum((data.obs[m]!=i)& (data.obs[groups_obs]==i)))

        TP=sum(TP_l)
        TN=sum(TN_l)
        FP=sum(FP_l)
        FN=sum(FN_l)

        SE.append(TP/(TP+FN))
        SP.append(TN/(TN+FP))
        PR.append(TP/(TP+FP))
        ACC.append((TN+TP)/(TN+TP+FN+FP))
        F1.append((2*TP)/(2*TP+FN+FP))
        report[m]= np.array([SE,SP,PR,ACC,F1]).flat

    report=pd.DataFrame(report)
    report.index=['SE', 'SP', 'PR', 'ACC', 'F1' ]
    report=report.transpose()
    return report

def grouped_classification_metrics(data, classification_obs, groups_obs):
    """ 
    Computes the main metrics of classification by comparing labels of cells classified with a given method (method of interest) and labels assigned with a different one (reference method) for each group defined by the latter.
    NB: labels must be correspondent!
           
    Parameters
    ----------
     
    data: anndata.AnnData
        an AnnData object.
        
    classification_obs: str
        a string specifying the AnnData.obs columns where the labels assigned by the method of interest are stored.
    groups_obs: str
        a string specifying the AnnData.obs where the labels assigned by the reference method.      
    
    Returns
    -------
    
    report: pandas.DataFrame
        a pandas.DataFame containing the per-group sensitivity (SE), specificity (SP), precision (PR), accuracy (ACC) and F1-score (F1) of the selected classification method.
    """
 
    report={}
    SE=[]
    SP=[]
    PR=[]
    ACC=[]
    F1=[]
    TP_l=[]
    TN_l=[]
    FP_l=[]
    FN_l=[]

    for i in data.obs[groups_obs].cat.categories:
        TP_l.append(sum((data.obs[classification_obs]==i)& (data.obs[groups_obs]==i)))
        TN_l.append(sum((data.obs[classification_obs]!=i)& (data.obs[groups_obs]!=i)))
        FP_l.append(sum((data.obs[classification_obs]==i)& (data.obs[groups_obs]!=i)))
        FN_l.append(sum((data.obs[classification_obs]!=i)& (data.obs[groups_obs]==i)))
    
    TP_l=np.array(TP_l)
    TN_l=np.array(TN_l)
    FP_l=np.array(FP_l)
    FN_l=np.array(FN_l)
        
    SE=TP_l/(TP_l+FN_l)
    SP=TN_l/(TN_l+FP_l)
    PR=TP_l/(TP_l+FP_l)
    ACC=(TN_l+TP_l)/(TN_l+TP_l+FN_l+FP_l)
    F1=(2*TP_l)/(2*TP_l+FN_l+FP_l)
    report= np.array([SE,SP,PR,ACC,F1])
    report=pd.DataFrame(report)
    report.index=['SE', 'SP', 'PR', 'ACC', 'F1' ]
    report.columns= data.obs[groups_obs].cat.categories
    report=report.transpose()
    return report

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
    data : anndata.AnnData
        An AnnData object containing the cell classification data.
    classification_obs : str
        A string specifying the AnnData.obs column where the labels assigned by the method of interest are stored.
    groups_obs : str
        A string specifying the AnnData.obs column where the labels assigned by the reference method are stored.
    columns_order : list of str, optional
        A list of strings specifying the order of columns in the heatmap.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap for the heatmap. Defaults to 'Reds'.
    save : str, optional
        A filename to save the heatmap. If provided, the heatmap is saved in the 'figures' directory with 'CIA_' prefix.

    Returns
    -------
    matplotlib.axes.Axes or None
        A heatmap AxesSubplot object is returned if `save` is None. Otherwise, the plot is saved to a file, and None is returned.

    Examples
    --------
    >>> group_composition(adata, 'method_labels', 'reference_labels')
    """
    # Compute the cross-tabulation of group memberships
    df = pd.crosstab(data.obs[groups_obs], data.obs[classification_obs])
    df = round((df / np.array(df.sum(axis=1)).reshape(len(df.index), 1)) * 100, 2)

    # Reorder columns if specified
    if columns_order:
        df = df.reindex(columns=columns_order)

    # Plot heatmap
    heatmap = sns.heatmap(df, cmap=cmap, annot=True)

    # Save the figure if `save` is provided
    if save:
        if not os.path.exists('./figures'):
            os.makedirs('figures')
        fig = heatmap.get_figure()
        fig.savefig(f"figures/CIA_{save}")
        plt.close(fig)  # Close the figure to prevent display in notebook environments
        return None

    return heatmap

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats
import itertools
import os

def grouped_distributions(data, columns_obs, groups_obs, cmap='Reds', scale_medians=None, save=None):
    """
    Plots a heatmap of median values for selected columns in AnnData.obs across cell groups and performs statistical tests to evaluate the differences in distributions between these groups. 
    A two-sided Wilcoxon test is conducted for each cell group to assess whether the distribution with the highest median significantly differs from the others.
    Additionally, for each selected column in AnnData.obs, a two-sided Mann-Whitney U test is carried out to determine if the distribution in the cell group with the highest median is significantly different from those in the other groups. 
    If a test fails (p >= 0.01), a message is printed.
    
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data.
    columns_obs : list of str
        Column names in AnnData.obs where the values of interest are stored.
    groups_obs : str
        Column name in AnnData.obs where the cell group labels are stored.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap for the heatmap. Defaults to 'Reds'.
    scale_medians : str, optional
        How to scale the median values in the heatmap. Options: 'row-wise', 'column-wise', or None.
    save : str, optional
        Filename to save the heatmap. If provided, saves the heatmap in 'figures' directory with 'CIA_' prefix.

    Returns
    -------
    None or AxesSubplot
        If `save` is provided, the heatmap is saved and None is returned. Otherwise, returns the AxesSubplot object.
   
    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad('your_data_file.h5ad')  # Load your AnnData file
    >>> adata.obs['group'] = ['A', 'B', 'A', 'B']  # Example group labels
    >>> adata.obs['column_obs1'] = [1, 2, 3, 4]  # Example feature values
    >>> adata.obs['column_obs2'] = [4, 3, 2, 1]  # Another example feature values
    >>> grouped_distributions(adata, ['column_obs1', 'column_obs2'], 'group', cmap='viridis')
    """

    # Compute the median values across the groups and select specified columns
    grouped_df = data.obs.groupby(groups_obs)[columns_obs].median()

    # Scale the medians if required
    if scale_medians == 'row-wise':
        grouped_df = grouped_df.div(grouped_df.sum(axis=1), axis=0)
    elif scale_medians == 'column-wise':
        grouped_df = grouped_df.div(grouped_df.sum(axis=0), axis=1)

    # Statistical analysis
    print('Performing Wilcoxon test on each cell group ...')
    for group in data.obs[groups_obs].unique():
        group_data = data[data.obs[groups_obs] == group]
        for comb in itertools.combinations(columns_obs, 2):
            x, y = group_data.obs[comb[0]], group_data.obs[comb[1]]
            if len(x) > 0 and len(y) > 0:
                stat, p_value = scipy.stats.wilcoxon(x, y, alternative='two-sided')
                if p_value >= 0.01:
                    print(f'Non-significant difference detected between {comb[0]} and {comb[1]} in group {group}.')

    print('Performing Mann-Whitney U test on each selected AnnData.obs column ...')
    combs = list(itertools.permutations(data.obs[groups_obs].unique(), 2))
    for column in columns_obs:
        for group_comb in combs:
            x, y = data[data.obs[groups_obs] == group_comb[0]].obs[column], data[data.obs[groups_obs] == group_comb[1]].obs[column]
            if len(x) > 0 and len(y) > 0:
                stat, p_value = scipy.stats.mannwhitneyu(x, y, alternative='two-sided')
                if p_value >= 0.01:
                    print(f'WARNING in {column} distribution: values in {group_comb[0]} group are not significantly different from values in {group_comb[1]} group (p= {p_value:.3f}).')

    # Plotting the heatmap
    heatmap = sns.heatmap(grouped_df, cmap=cmap, annot=True)

    # Save the heatmap if the filename is provided
    if save:
        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        heatmap.get_figure().savefig(f"./figures/CIA_{save}")
        plt.close(heatmap.get_figure())  # Close the figure to free up memory
        return None

    return heatmap



def classification_metrics(data, classification_obs, groups_obs):
    """
    Computes the main metrics of classification by comparing labels of cells classified with given methods 
    (methods of interest) to labels assigned with a different one (reference method).
    NB: labels must be correspondent!
           
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data.
    classification_obs : list of str
        A list of strings specifying the AnnData.obs columns where the labels assigned by the methods of interest are stored.
    groups_obs : str
        A string specifying the AnnData.obs column where the labels assigned by the reference method are stored.      
    
    Returns
    -------
    report : pandas.DataFrame
        A pandas.DataFrame containing the overall sensitivity (SE), specificity (SP), precision (PR), 
        accuracy (ACC), and F1-score (F1) of the selected classification methods.
    
    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad('your_data_file.h5ad')  # Load your AnnData file
    >>> adata.obs['method1'] = ['label1', 'label2', 'label1', 'label2']  # Example classification
    >>> adata.obs['method2'] = ['label1', 'label1', 'label2', 'label2']  # Another example classification
    >>> adata.obs['reference'] = ['label1', 'label1', 'label2', 'label2']  # Reference classification
    >>> classification_metrics(adata, ['method1', 'method2'], 'reference')
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
    Computes the main metrics of classification for each group defined by the reference method,
    comparing the labels from the method of interest with the reference labels.
    NB: Corresponding labels are assumed between methods!
           
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data.
    classification_obs : str
        The AnnData.obs column where the labels assigned by the method of interest are stored.
    groups_obs : str
        The AnnData.obs column where the labels assigned by the reference method are stored.      
    
    Returns
    -------
    report : pandas.DataFrame
        A DataFrame containing the per-group sensitivity (SE), specificity (SP), precision (PR),
        accuracy (ACC), and F1-score (F1) of the selected classification method.

    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad('your_data_file.h5ad')  # Load your AnnData file
    >>> classification_obs = 'predicted_labels'  # Example classification column
    >>> groups_obs = 'actual_labels'  # Reference classification column
    >>> metrics_report = grouped_classification_metrics(adata, classification_obs, groups_obs)
    >>> print(metrics_report)
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

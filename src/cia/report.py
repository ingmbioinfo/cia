import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools 
import scipy
import os


def group_composition(data, classification_obs, ref_obs, columns_order=None, cmap='Reds', save=None):
    """
    Plots a heatmap showing the percentages of cells classified with a given method (method of interest) in cell groups defined with a different one (reference method).

    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell classification data.
    classification_obs : str
        A string specifying the AnnData.obs column where the labels assigned by the method of interest are stored.
    ref_obs : str
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
    df = pd.crosstab(data.obs[ref_obs], data.obs[classification_obs])
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

def grouped_distributions(data, columns_obs, ref_obs, cmap='Reds', scale_medians=None, save=None):
    """
    Plots a heatmap of median values for selected columns in AnnData.obs across cell groups and performs statistical tests 
    to evaluate the differences in distributions. The Wilcoxon test checks if each group's signature score is significantly 
    higher than others in the same group. The Mann-Whitney U test checks if each signature has the highest score values 
    in the corresponding group compared to all other groups.
    
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data.
    columns_obs : list of str
        Column names in AnnData.obs where the values of interest are stored.
    ref_obs : str
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
    """

    grouped_df=data.obs.groupby(ref_obs).median()
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
    for i in data.obs[ref_obs].cat.categories:
        subsets[i]= data[data.obs[ref_obs]==i].obs[columns_obs]
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
    combs=list(itertools.permutations(data.obs[ref_obs].cat.categories,2))
    count=0
    for i in columns_obs:
        sign={}
        l=[]
        for c in data.obs[ref_obs].cat.categories:
            l.append(subsets[c][i].values)
        sign[i]=pd.DataFrame(l).transpose()
        sign[i].columns=data.obs[ref_obs].cat.categories
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





def compute_classification_metrics(data, classification_obs, ref_obs, unassigned_label=''):
    """
    Computes the main metrics of classification by comparing labels of cells classified with given methods 
    (methods of interest) to labels assigned with a different one (reference method).
    Cells labeled as unassigned_label in any method of interest are excluded from the metrics calculation.
    Additionally, if present, the percentage of unassigned cells for each classification method is calculated and reported.
           
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data.
    classification_obs : list of str
        A list of strings specifying the AnnData.obs columns where the labels assigned by the methods of interest are stored.
    ref_obs : str
        A string specifying the AnnData.obs column where the labels assigned by the reference method are stored.
    unassigned_label : str, optional
        The label used to mark unassigned cells in the classification columns. Cells with this label will be excluded 
        from the metrics calculation. Default is an empty string, which means no cells are excluded based on their label.

    Returns
    -------
    report : pandas.DataFrame
        A pandas.DataFrame containing the overall sensitivity (SE), specificity (SP), precision (PR), 
        accuracy (ACC), F1-score (F1), and, if specified, the percentage of unassigned cells (%UN) for each classification method 
        compared to the reference method.
    
    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad('your_data_file.h5ad')  # Load your AnnData file
    >>> adata.obs['method1'] = ['label1', 'label2', 'label1', 'label2']  # Example classification
    >>> adata.obs['method2'] = ['label1', 'label1', 'label2', 'label2']  # Another example classification
    >>> adata.obs['reference'] = ['label1', 'label1', 'label2', 'label2']  # Reference classification
    >>> classification_metrics(adata, ['method1', 'method2'], 'reference', unassigned_label='Unassigned')
    """
    report = {}
    
    for m in classification_obs:
        SE, SP, PR, ACC, F1, UN = [], [], [], [], [], []

        total_cells = len(data.obs)
        unassigned_count = sum(data.obs[m] == unassigned_label)
        UN.append(round((unassigned_count / total_cells) * 100,2))  # Calculate percentage of unassigned cells

        filtered_data = data[data.obs[m] != unassigned_label]

        TP_l, TN_l, FP_l, FN_l = [], [], [], []

        for i in filtered_data.obs[ref_obs].cat.categories:
            TP_l.append(sum((filtered_data.obs[m] == i) & (filtered_data.obs[ref_obs] == i)))
            TN_l.append(sum((filtered_data.obs[m] != i) & (filtered_data.obs[ref_obs] != i)))
            FP_l.append(sum((filtered_data.obs[m] == i) & (filtered_data.obs[ref_obs] != i)))
            FN_l.append(sum((filtered_data.obs[m] != i) & (filtered_data.obs[ref_obs] == i)))

        TP = sum(TP_l)
        TN = sum(TN_l)
        FP = sum(FP_l)
        FN = sum(FN_l)

        SE.append(TP / (TP + FN))
        SP.append(TN / (TN + FP))
        PR.append(TP / (TP + FP))
        ACC.append((TN + TP) / (TN + TP + FN + FP))
        F1.append((2 * TP) / (2 * TP + FN + FP))

        metrics = np.array([SE, SP, PR, ACC, F1, UN]).flat
        report[m] = metrics

    report = pd.DataFrame(report)
    report.index = ['SE', 'SP', 'PR', 'ACC', 'F1', '%UN']
    report = report.transpose()
    if sum(report['%UN'])==0:
        del report['%UN']
    return report

def grouped_classification_metrics(data, classification_obs, ref_obs, unassigned_label=''):
    """
    Computes the main metrics of classification for each group defined by the reference method,
    comparing the labels from the method of interest with the reference labels. Additionally,
    if specified, computes the percentage of unlabelled cells for each group.
           
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data.
    classification_obs : str
        The AnnData.obs column where the labels assigned by the method of interest are stored.
    ref_obs : str
        The AnnData.obs column where the labels assigned by the reference method are stored.
    
    Returns
    -------
    report : pandas.DataFrame
        A DataFrame containing the per-group sensitivity (SE), specificity (SP), precision (PR),
        accuracy (ACC), F1-score (F1), and if present, the percentage of unassigned cells (%UN) for the selected classification method.
    
    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad('your_data_file.h5ad')  # Load your AnnData file
    >>> classification_obs = 'predicted_labels'
    >>> ref_obs = 'actual_labels'
    >>> metrics_report = grouped_classification_metrics(adata, classification_obs, ref_obs)
    """
    report = []

    for group in data.obs[ref_obs].cat.categories:
        is_group = data.obs[ref_obs] == group
        is_unassigned = data.obs[classification_obs] == unassigned_label

        TP = np.sum(data.obs[classification_obs][is_group] == group)
        TN = np.sum(data.obs[classification_obs][~is_group] != group)
        FP = np.sum(data.obs[classification_obs][~is_group] == group)
        FN = np.sum(data.obs[classification_obs][is_group] != group)

        SE = TP / (TP + FN) if TP + FN else 0
        SP = TN / (TN + FP) if TN + FP else 0
        PR = TP / (TP + FP) if TP + FP else 0
        ACC = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0
        F1 = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0

        # Calculate the percentage of unassigned cells for this group
        group_total = np.sum(is_group)
        unassigned_count = np.sum(is_unassigned & is_group)
        UN = (unassigned_count / group_total) * 100 if group_total else 0

        report.append([SE, SP, PR, ACC, F1, UN])

    report_df = pd.DataFrame(report, columns=['SE', 'SP', 'PR', 'ACC', 'F1', '%UN'], index=data.obs[ref_obs].cat.categories)
    if sum(report_df['%UN'])==0:
        del report_df['%UN']
    return report_df
    
def plot_group_composition(df, ref_col, comp_col, plot_type='percentage', palette='Set3', show_legend=True):
    """
    Plot the composition of each reference group as a horizontal stacked bar plot.
    The composition can be shown either as raw counts or as percentages.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the data to be plotted.
    ref_col : str 
        the name of the column representing the reference grouping variable.
    comp_col: str
        the name of the column representing the grouping to be compared.
    plot_type : str
        indicates whether to plot 'percentage' or 'raw' counts. Defaults to 'percentage'.
    palette : str or list 
        the color palette to use. Defaults to 'Set3'.
    show_legend : bool
        whether to display the legend on the plot. Defaults to True.
    
    Returns
    -------
    AxesSubplot
    """
    
    # Check if specified columns exist in the DataFrame
    if not {ref_col, comp_col}.issubset(df.columns):
        raise ValueError("Specified columns are not in the DataFrame")

    # Create a contingency table of counts
    contingency_table = pd.crosstab(df[ref_col], df[comp_col], dropna=False)
    
    # Ensure all groups are represented, even with zero counts
    all_groups = list(contingency_table.columns)

    # Calculate percentages if required
    if plot_type == 'percentage':
        contingency_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

    # Set up the color palette
    colors = sns.color_palette(palette, n_colors=len(all_groups))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = None
    for i, group in enumerate(all_groups):
        values = contingency_table[group].values
        ax.barh(contingency_table.index, values, left=bottom, label=group, color=colors[i])
        if bottom is None:
            bottom = values
        else:
            bottom += values

    ax.set_xlabel('Percentage' if plot_type == 'percentage' else 'Count')
    ax.set_ylabel(ref_col)
    ax.set_title('Group Composition by ' + ref_col)

    if show_legend:
        ax.legend(title=comp_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()


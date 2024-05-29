CIA BENCHMARKING
================

Description
-----------

In this notebook we compared **CIA** with **other classification tools**
among the newest and all compatible with Scanpy workflow: - **Scorect**
(`Github public repository <https://github.com/LucasESBS/scoreCT>`__),
which requires a gmt file with signatures and a test dataset already
divided into clusters with Differentially Expressed Genes (DEGs) for
each of them. It relies on the division of **ranked DEGs into bins** and
the calculation of a **weighted score** for each cluster depending on
how many DEGs of each bin are also present in the gmt signatures. -
**Celltypist** (Domínguez et al., Science, 2022
`[1] <https://www.science.org/doi/10.1126/science.abl5197>`__) which,
exploiting a **combined approach based on Logistic Regression and
Stocastic Gradient Descent**, extracts the most revelevant feature of
**each cluster of the training set** and **trains a classification
model**. - **Besca** auto_annot module (Mädler et al., NAR Genomics and
Bioinformatics, 2021 `[2] <https://doi.org/10.1093/nargab/lqab102>`__),
which can be run with **Logistic Regression** (LR) or **Support Vector
Machine** (SVM) algorithms.

.. code:: ipython3

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import seaborn as sns
    import multiprocessing
    from functools import partial
    from scipy.sparse import issparse
    from scipy import sparse
    import time
    from sklearn import metrics
    from scipy import sparse
    from cia import investigate, report, external
    import pickle
    import celltypist
    import scorect as ct
    import besca as bc


.. parsed-literal::

    Global seed set to 0


Training dataset
----------------

We **trained all the classifiers** with Hao et al 2021
`[3] <https://www.sciencedirect.com/science/article/pii/S0092867421005833>`__
**PBMC atlas**, the same from which we extracted the gene signatures in
`CIA workflow
tutorial <../workflow/Cluster_Independent_Annotation.ipynb>`__.

.. code:: ipython3

    !cp ./data/atlas_final.h5ad.gz  ./data/copy_atlas.h5ad.gz
    !gunzip ./data/copy_atlas.h5ad.gz

.. code:: ipython3

    # to read the atlas data
    adata= sc.read('./data/copy_atlas.h5ad')
    adata




.. parsed-literal::

    AnnData object with n_obs × n_vars = 154975 × 1555
        obs: 'nCount_ADT', 'nFeature_ADT', 'nCount_RNA', 'nFeature_RNA', 'orig.ident', 'lane', 'donor', 'time', 'celltype.l1', 'celltype.l2', 'celltype.l3', 'Phase', 'nCount_SCT', 'nFeature_SCT', 'Cell type', 'B', 'CD4 T', 'CD8 T', 'DC', 'Mono', 'NK', 'Platelet', 'CD8 T_negative', 'CD4 T_negative', 'Prediction fast mode', 'Prediction standard mode', 'Prediction q', 'Prediction p-val', 'B_filtered_FC', 'CD4 T_filtered_FC', 'CD8 T_filtered_FC', 'DC_filtered_FC', 'Mono_filtered_FC', 'NK_filtered_FC', 'Platelet_filtered_FC'
        var: 'features', 'highly_variable', 'means', 'dispersions', 'dispersions_norm'
        uns: 'Cell type_colors', 'Prediction fast mode_colors', 'Prediction p-val_colors', 'Prediction q_colors', 'Prediction standard mode_colors', 'celltype.l1_colors', 'celltype.l2_colors', 'celltype.l3_colors', 'hvg', 'neighbors', 'signature_based_classification'
        obsm: 'X_apca', 'X_aumap', 'X_pca', 'X_spca', 'X_umap', 'X_wnn.umap'
        varm: 'PCs', 'SPCA'
        obsp: 'distances'



.. code:: ipython3

    !rm ./data/copy_atlas.h5ad

For standardization purposes and because of **Celltypist strict
requirements**, we rescaled counts from 0 to 10000 for each cell and we
log transformed the resulting values, accordingly with `Scanpy
tutorial <https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html>`__.

.. code:: ipython3

    sc.set_figure_params(scanpy=True, dpi=100)

.. code:: ipython3

    sc.pl.umap(adata, color='Cell type')



.. image:: output_11_0.png
   :width: 463px
   :height: 371px


Test dataset
------------

We **fitted the trained model of each classifier on**
`PBMC3K <https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html>`__
from Satija et al. 2015
`[4] <https://www.nature.com/articles/nbt.3192>`__, and we compared the
performances of classification with CIA ones.

NB: the test dataset was normalized in the very same way of the training
dataset.

.. code:: ipython3

    pbmc3k=sc.read('data/pbmc3k_classified.h5ad')

Cluster-based annotation
------------------------

First of all, we **challanged cluster-dependent classifiers**, which
requires that the test dataset must be already divided into meaningful
clusters by setting the proper resolution.

Scorect
~~~~~~~

.. code:: ipython3

    # To read gmt file  (same signatures used for CIA)
    ref_marker = ct.read_markers_from_file('data/atlas.gmt')
    
    # To perform differential expression analysis and obtain DEGs
    sc.tl.rank_genes_groups(pbmc3k, groupby='Cell type', n_genes=len(pbmc3k.raw.var), use_raw=True)
    marker_df = ct.wrangle_ranks_from_anndata(pbmc3k)
    
    
    # To set the background genes - here, all the genes used to run the differential gene expression test
    background = pbmc3k.raw.var.index.tolist()
    
    # to score cell types for each cluster 
    ct_pval, ct_score = ct.celltype_scores(nb_bins=5,
                                            ranked_genes=marker_df,
                                            K_top = 300,
                                            marker_ref=ref_marker,
                                            background_genes=background)


.. parsed-literal::

    WARNING: Default of the method has been changed to 't-test' from 't-test_overestim_var'
    Wrangling: Number of markers used in ranked_gene_groups:  13714
    Wrangling: Groups used for ranking: Cell type


.. code:: ipython3

    # To assign identity to each cluster depending on the scores
    pbmc3k.obs['Prediction scorect'] = ct.assign_celltypes(cluster_assignment=pbmc3k.obs['Cell type'], ct_pval_df=ct_pval, ct_score_df=ct_score, cutoff=0.1)
    sc.pl.umap(pbmc3k, color=['Cell type', 'Prediction fast mode', 'Prediction p-val','Prediction scorect'], wspace=0.4)


.. parsed-literal::

    ... storing 'Prediction scorect' as categorical



.. image:: output_20_1.png
   :width: 1907px
   :height: 375px


Performances
~~~~~~~~~~~~

We next exploited **classification_metrics** function of **CIA report
module** to evaluate the performance of each classifier using Satija et
al. 2015 `[4] <https://www.nature.com/articles/nbt.3192>`__ annotation
as ground truth.

.. code:: ipython3

    report.classification_metrics(pbmc3k, classification_obs=['Prediction fast mode', 'Prediction p-val', 'Prediction scorect'], groups_obs='Cell type')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>SE</th>
          <th>SP</th>
          <th>PR</th>
          <th>ACC</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Prediction fast mode</th>
          <td>0.895375</td>
          <td>0.982563</td>
          <td>0.895375</td>
          <td>0.970107</td>
          <td>0.895375</td>
        </tr>
        <tr>
          <th>Prediction p-val</th>
          <td>0.847991</td>
          <td>0.980035</td>
          <td>0.876224</td>
          <td>0.961172</td>
          <td>0.861876</td>
        </tr>
        <tr>
          <th>Prediction scorect</th>
          <td>0.866187</td>
          <td>0.977698</td>
          <td>0.866187</td>
          <td>0.961768</td>
          <td>0.866187</td>
        </tr>
      </tbody>
    </table>
    </div>



Despite the good performances of Scorect (F1-score > 0.85, with overall
performances comparable with CIA predictions), 2 whole clusters were
misclassified, meaning that **2 cell populations were not detected**.

Cell level classification
-------------------------

Since, like CIA, **Celltypist and Besca** allow to **automatically
classify datasets cell-by-cell**, without requiring the clustering, we
challanged both classifiers at single cell level.

Celltypist
~~~~~~~~~~

.. code:: ipython3

    # To train Celltypist with PBMC atlas feature and ground truth labels ('Cell type' obs).
    
    # COMPUTATIONALLY HEAVY! results have been stored in 'data/model_atlas.pkl'.
    #new_model = celltypist.train(adata, labels = 'Cell type', n_jobs = 32, feature_selection = True)
    
    # To load trained model
    new_model=celltypist.Model.load('data/model_atlas_no_other_t.pkl')
    new_model




.. parsed-literal::

    CellTypist model with 7 cell types and 1280 features
        date: 2022-05-26 13:54:42.584144
        cell types: B, CD4 T, ..., Platelet
        features: TTLL10, TNFRSF18, ..., MTRNR2L8



.. code:: ipython3

    # To fit the model and predict pbmc3k cell identities.
    predictions = celltypist.annotate(pbmc3k.raw.to_adata(), model = new_model)


.. parsed-literal::

    🔬 Input data has 2638 cells and 13714 genes
    🔗 Matching reference genes in the model
    🧬 1157 features used for prediction
    ⚖️ Scaling input data
    🖋️ Predicting labels
    ✅ Prediction done!


.. code:: ipython3

    pbmc3k.obs['Prediction celltypist']=predictions.predicted_labels

.. code:: ipython3

    sc.pl.umap(pbmc3k, color=['Cell type', 'Prediction fast mode','Prediction p-val','Prediction celltypist'], wspace=0.5)


.. parsed-literal::

    ... storing 'Prediction celltypist' as categorical



.. image:: output_31_1.png
   :width: 2034px
   :height: 375px


Besca
~~~~~

.. code:: ipython3

    # to train Besca Logistic Regression model and perform feature selection
    adata_train, adata_test_corrected = bc.tl.auto_annot.merge_data([adata], pbmc3k, genes_to_use = 'all', merge = 'scanorama')


.. parsed-literal::

    merging with scanorama
    using scanorama rn
    Found 350 genes among all datasets
    [[0.         0.94427597]
     [0.         0.        ]]
    Processing datasets (0, 1)
    integrating training set
    calculating intersection


.. code:: ipython3

    classifier, scaler = bc.tl.auto_annot.fit(adata_train, method='logistic_regression', celltype='Cell type', njobs=32)
    bc.tl.auto_annot.adata_predict(classifier = classifier, adata_pred = adata_test_corrected, adata_orig=pbmc3k,
                                   threshold =0.1, scaler=scaler)
    pbmc3k.obs['Prediction besca LR']=pbmc3k.obs['auto_annot']
    del pbmc3k.obs['auto_annot']


.. parsed-literal::

    [Parallel(n_jobs=32)]: Using backend LokyBackend with 32 concurrent workers.
    [Parallel(n_jobs=32)]: Done   2 out of   5 | elapsed:  2.6min remaining:  4.0min
    [Parallel(n_jobs=32)]: Done   5 out of   5 | elapsed:  2.9min finished


.. code:: ipython3

    # to train Besca SVM model and perform feature selection
    
    # COMPUTATIONALLY HEAVY, results of this chunk have been already saved in pbmc3k.obs['auto_annot']
    
    # classifier, scaler = bc.tl.auto_annot.fit(adata_train, method='linear', celltype='Cell type', njobs=10)
    # bc.tl.auto_annot.adata_predict(classifier = classifier, adata_pred = adata_test_corrected, adata_orig=pbmc3k, 
    #                                threshold =0.1, scaler=scaler)
    # pbmc3k.obs['Prediction besca SVM']=pbmc3k.obs['auto_annot']
    # del pbmc3k.obs['auto_annot']

.. code:: ipython3

    sc.pl.umap(pbmc3k, color=['Cell type','Prediction fast mode','Prediction p-val', 'Prediction besca LR','Prediction besca SVM'], wspace=0.5)


.. parsed-literal::

    ... storing 'Prediction besca LR' as categorical



.. image:: output_36_1.png
   :width: 2034px
   :height: 723px


Performances
~~~~~~~~~~~~

.. code:: ipython3

    report.classification_metrics(pbmc3k, classification_obs=['Prediction fast mode',
                                                             'Prediction p-val', 'Prediction celltypist',
                                                             'Prediction besca LR','Prediction besca SVM'], groups_obs='Cell type')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>SE</th>
          <th>SP</th>
          <th>PR</th>
          <th>ACC</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Prediction fast mode</th>
          <td>0.895375</td>
          <td>0.982563</td>
          <td>0.895375</td>
          <td>0.970107</td>
          <td>0.895375</td>
        </tr>
        <tr>
          <th>Prediction p-val</th>
          <td>0.847991</td>
          <td>0.980035</td>
          <td>0.876224</td>
          <td>0.961172</td>
          <td>0.861876</td>
        </tr>
        <tr>
          <th>Prediction celltypist</th>
          <td>0.775588</td>
          <td>0.962598</td>
          <td>0.775588</td>
          <td>0.935882</td>
          <td>0.775588</td>
        </tr>
        <tr>
          <th>Prediction besca LR</th>
          <td>0.845337</td>
          <td>0.974223</td>
          <td>0.845337</td>
          <td>0.955811</td>
          <td>0.845337</td>
        </tr>
        <tr>
          <th>Prediction besca SVM</th>
          <td>0.868840</td>
          <td>0.978140</td>
          <td>0.868840</td>
          <td>0.962526</td>
          <td>0.868840</td>
        </tr>
      </tbody>
    </table>
    </div>



Notably, **CIA fast mode prediction was the best one**, followed by the
computationally heavier Besca SVM prediction and CIA p-val mode (which
are comparable).

Over-clustering driven approach
-------------------------------

**Celltypist** has an interesting feature called **‘majority voting’**,
which consist of the refinement of cell identities within local
subclusters after an **over-clustering step**. Basically, **for each
subcluster**, the **label of the most abundant cell type is extended**
to the whole cell group. This is an **additional step that goes beyond
the CIA workflow**, but since we wanted to compare the results at the
best possible conditions of classification, we wrote
**celltypist_majority_vote** function (**external module**) to reproduce
the ‘majority voting’ approach.

To be more clear, **each of the cell-level classifications were matched
with those 68 clusters** (small groups of very similar cells) and at
each mini-cluster was assigned the most abundat cell type label.

.. code:: ipython3

    sc.pl.umap(pbmc3k, color='leiden_5')



.. image:: output_43_0.png
   :width: 567px
   :height: 622px


Celltypist majority voting
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # To run Celltypist with majority voting
    predictions = celltypist.annotate(pbmc3k.raw.to_adata(), model = new_model, majority_voting=True)


.. parsed-literal::

    🔬 Input data has 2638 cells and 13714 genes
    🔗 Matching reference genes in the model
    🧬 1157 features used for prediction
    ⚖️ Scaling input data
    🖋️ Predicting labels
    ✅ Prediction done!
    👀 Can not detect a neighborhood graph, will construct one before the over-clustering
    ⛓️ Over-clustering input data with resolution set to 5
    🗳️ Majority voting the predictions
    ✅ Majority voting done!


.. code:: ipython3

    pbmc3k.obs['Prediction celltypist majority voting']=predictions.predicted_labels['majority_voting']

.. code:: ipython3

    sc.pl.umap(pbmc3k, color=['Cell type','Prediction celltypist majority voting'], wspace=0.6)



.. image:: output_47_0.png
   :width: 1134px
   :height: 375px


Majority voting embedded in CIA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # To replicate the majority voting step on CIA annotations exploiting celltypist_majority_vote.
    colnames=['Prediction fast mode', 'Prediction p-val']
    external.celltypist_majority_vote(pbmc3k,classification_obs=colnames)


.. parsed-literal::

    Reference annotation not selected.
    Computing over-clustering with leiden algorithm (resolution= 5) ...
    Dataset has been divided into 69 groups accordingly with trascriptional similarities.
    
    Over-clustering result saved in AnnData.obs["leiden_5"].
    Extending the more represented cell type label to each cell group...
    
    New classification labels have been stored in AnnData.obs["Prediction fast mode majority voting"]. 
    
    New classification labels have been stored in AnnData.obs["Prediction p-val majority voting"]. 
    


.. code:: ipython3

    colnames_mv=['Cell type','Prediction fast mode majority voting', 'Prediction p-val majority voting']

.. code:: ipython3

    sc.pl.umap(pbmc3k, color=colnames_mv, wspace=0.6)



.. image:: output_51_0.png
   :width: 1648px
   :height: 375px


Majority voting applied to Besca
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    colnames=['Prediction besca LR','Prediction besca SVM']
    external.celltypist_majority_vote(pbmc3k,classification_obs=colnames)


.. parsed-literal::

    Reference annotation not selected.
    Computing over-clustering with leiden algorithm (resolution= 5) ...
    Dataset has been divided into 69 groups accordingly with trascriptional similarities.
    
    Over-clustering result saved in AnnData.obs["leiden_5"].
    Extending the more represented cell type label to each cell group...
    
    New classification labels have been stored in AnnData.obs["Prediction besca LR majority voting"]. 
    
    New classification labels have been stored in AnnData.obs["Prediction besca SVM majority voting"]. 
    


.. code:: ipython3

    colnames_mv=['Cell type','Prediction besca LR majority voting', 'Prediction besca SVM majority voting']

.. code:: ipython3

    sc.pl.umap(pbmc3k, color=colnames_mv, wspace=0.6)



.. image:: output_55_0.png
   :width: 1648px
   :height: 375px


Performances
~~~~~~~~~~~~

.. code:: ipython3

    report.classification_metrics(pbmc3k, classification_obs=['Prediction celltypist majority voting', 'Prediction fast mode majority voting', 'Prediction p-val majority voting', 
                 'Prediction besca LR majority voting','Prediction besca SVM majority voting'], groups_obs='Cell type')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>SE</th>
          <th>SP</th>
          <th>PR</th>
          <th>ACC</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Prediction celltypist majority voting</th>
          <td>0.921531</td>
          <td>0.986922</td>
          <td>0.921531</td>
          <td>0.977580</td>
          <td>0.921531</td>
        </tr>
        <tr>
          <th>Prediction fast mode majority voting</th>
          <td>0.966262</td>
          <td>0.994377</td>
          <td>0.966262</td>
          <td>0.990361</td>
          <td>0.966262</td>
        </tr>
        <tr>
          <th>Prediction p-val majority voting</th>
          <td>0.966262</td>
          <td>0.994377</td>
          <td>0.966262</td>
          <td>0.990361</td>
          <td>0.966262</td>
        </tr>
        <tr>
          <th>Prediction besca LR majority voting</th>
          <td>0.939348</td>
          <td>0.989891</td>
          <td>0.939348</td>
          <td>0.982671</td>
          <td>0.939348</td>
        </tr>
        <tr>
          <th>Prediction besca SVM majority voting</th>
          <td>0.957544</td>
          <td>0.992924</td>
          <td>0.957544</td>
          <td>0.987870</td>
          <td>0.957544</td>
        </tr>
      </tbody>
    </table>
    </div>



Notably, **all the classifications** resulted to have a **very high
F1-score**. However, **CIA performed the best classifications** also
after the majoity voting step, followed by Besca SVM (comparable).

Conclusions
-----------

In this notebook we challenged the **newest classifiers compatible with
Scanpy** in annotating the `PBMC3K
dataset <https://scanpy.readthedocs.io/en/stable/generated/scanpy.datasets.pbmc3k.html>`__
and compared their perfomances with CIA ones. Here are summarized all
the results:

-  With **cluster-driven approach** (Scorect, at default setting) **it
   was impossible to detect 2 cell populations**, since for 2 clusters
   the top scored signature wasn’t the right one. This result
   highlighted that if the classification is cluster-dependent eventual
   misclassifications can be propagated to entire clusters.
-  Overall the **classifiers allowing cell-level annotation were able to
   correctly predict the identity of the majority of cells**, indicating
   that is possible to accurately annotate the datasets without the
   arbitrariety which characterizes the clustering-driven manual
   annotation. Among them **CIA resulted to be the best at cell-level
   annotation**.
-  **The winning approach** of this challenge **is cell-level automatic
   annotation followed by** over-clustering and **majority voting**.
   With this approach, clustering with very high resolution is performed
   only after cell-level classification. **When clustering is used to
   refine** an already accurate automatic labelling (and so less
   influenced by analysts arbitrary decisions), instead of completely
   drive the annotation, **results to be an effective way to integrate**
   the contributions of **marker genes expression with** the more
   general concept of **transcriptional similarity** in defining cell
   identity. Also with this approach **CIA obtained the best results**.

.. code:: ipython3

    columns=['Prediction fast mode','Prediction p-val','Prediction celltypist', 'Prediction besca LR','Prediction besca SVM',
            'Prediction fast mode majority voting','Prediction p-val majority voting','Prediction celltypist majority voting',
            'Prediction besca LR majority voting','Prediction besca SVM majority voting','Prediction scorect']

.. code:: ipython3

    report.classification_metrics(pbmc3k, classification_obs=columns, groups_obs='Cell type')




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>SE</th>
          <th>SP</th>
          <th>PR</th>
          <th>ACC</th>
          <th>F1</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Prediction fast mode</th>
          <td>0.895375</td>
          <td>0.982563</td>
          <td>0.895375</td>
          <td>0.970107</td>
          <td>0.895375</td>
        </tr>
        <tr>
          <th>Prediction p-val</th>
          <td>0.847991</td>
          <td>0.980035</td>
          <td>0.876224</td>
          <td>0.961172</td>
          <td>0.861876</td>
        </tr>
        <tr>
          <th>Prediction celltypist</th>
          <td>0.775588</td>
          <td>0.962598</td>
          <td>0.775588</td>
          <td>0.935882</td>
          <td>0.775588</td>
        </tr>
        <tr>
          <th>Prediction besca LR</th>
          <td>0.845337</td>
          <td>0.974223</td>
          <td>0.845337</td>
          <td>0.955811</td>
          <td>0.845337</td>
        </tr>
        <tr>
          <th>Prediction besca SVM</th>
          <td>0.868840</td>
          <td>0.978140</td>
          <td>0.868840</td>
          <td>0.962526</td>
          <td>0.868840</td>
        </tr>
        <tr>
          <th>Prediction fast mode majority voting</th>
          <td>0.966262</td>
          <td>0.994377</td>
          <td>0.966262</td>
          <td>0.990361</td>
          <td>0.966262</td>
        </tr>
        <tr>
          <th>Prediction p-val majority voting</th>
          <td>0.966262</td>
          <td>0.994377</td>
          <td>0.966262</td>
          <td>0.990361</td>
          <td>0.966262</td>
        </tr>
        <tr>
          <th>Prediction celltypist majority voting</th>
          <td>0.921531</td>
          <td>0.986922</td>
          <td>0.921531</td>
          <td>0.977580</td>
          <td>0.921531</td>
        </tr>
        <tr>
          <th>Prediction besca LR majority voting</th>
          <td>0.939348</td>
          <td>0.989891</td>
          <td>0.939348</td>
          <td>0.982671</td>
          <td>0.939348</td>
        </tr>
        <tr>
          <th>Prediction besca SVM majority voting</th>
          <td>0.957544</td>
          <td>0.992924</td>
          <td>0.957544</td>
          <td>0.987870</td>
          <td>0.957544</td>
        </tr>
        <tr>
          <th>Prediction scorect</th>
          <td>0.866187</td>
          <td>0.977698</td>
          <td>0.866187</td>
          <td>0.961768</td>
          <td>0.866187</td>
        </tr>
      </tbody>
    </table>
    </div>



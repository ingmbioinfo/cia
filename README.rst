.. image:: docs/logo.png
   :align: right
   :alt: 
   :width: 120

CIA (Cluster Independent Annotation)
====================================

CIA (Cluster Independent Annotation) is a cutting-edge computational tool designed to accurately classify cells in scRNA-seq datasets using gene signatures. This tool operates without the need for a fully annotated reference dataset or complex machine learning processes, providing a highly user-friendly and practical solution for cell type annotation.

Description
-----------

CIA synthesizes the information of each signature expression into a single score value for each cell. By comparing these score values, CIA assigns labels to each cell based on the top-scored signature. CIA can filter scores by their distribution or significance, allowing comparison of genesets with lengths spanning tens to thousands of genes.

CIA is implemented in both R and Python, making it compatible with all major single-cell analysis tools like SingleCellExperiment, Seurat, and Scanpy. This dual compatibility ensures seamless integration into existing workflows.

Key Features
------------

- **Automatic Annotation**: Accurately labels cell types in scRNA-seq datasets based on gene signatures.
- **Clustering-Free**: Operates independently of clustering steps, enabling flexible and rapid data exploration.
- **Multi-Language Support**: Available in both R and Python to suit diverse user preferences.
- **Compatibility**: Integrates with popular single-cell data formats (AnnData, SingleCellExperiment, SeuratObject).
- **Statistical Analysis**: Offers functions for evaluating the quality of signatures and classification performance.
- **Documentation and Tutorials**: Comprehensive guides to facilitate easy adoption and integration into existing workflows.

Documentation
------------------------------

- **Python Package**: `CIA Python <https://pypi.org/project/cia-python/>`_
- **Python docs**: `CIA Python documentation <https://cia-python.readthedocs.io/en/latest/index.html>`_
- **R Package and Tutorial**: `CIA R GitHub Repository <https://github.com/ingmbioinfo/CIA_R>`_

Installation
------------------------------
``cia`` package could be installed using pip:

.. code-block:: shell

	pip install cia-python

To install the github developing version run the following commands:

.. code-block:: shell

	git clone https://github.com/ingmbioinfo/cia.git

	cd cia

	pip install -e .



Citation
--------

If you use CIA in your work, please cite our publication as follows:

Ferrari I, Battistella M, Vincenti F, Gobbini A, Notarbartolo S, Costanza J, Biffo S, Grifantini R, Abrignani S, Galeota E. (2023). "CIA: a Cluster Independent Annotation method to investigate cell identities in scRNA-seq data". bioRxiv. doi: `10.1101/2023.11.30.569382 <https://doi.org/10.1101/2023.11.30.569382>`_.

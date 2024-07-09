.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/CIA.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/CIA
    .. image:: https://readthedocs.org/projects/CIA/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://CIA.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/CIA/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/CIA
    .. image:: https://img.shields.io/pypi/v/CIA.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/CIA/
    .. image:: https://img.shields.io/conda/vn/conda-forge/CIA.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/CIA
    .. image:: https://pepy.tech/badge/CIA/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/CIA
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/CIA

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

===
CIA
===


CIA (Cluster Independent Annotation) is a new computational tool that enables a highly accurate automatic classification of cells in scRNA-seq datasets exploiting gene signatures.

Given a collection of signatures, CIA synthesizes the information of each signature expression in a single score value for each cell. Comparing the score values, the package assigns labels to each cell accordingly with the top scored signature. This classifier allows the use of different modes, depending on whether the user prefers a faster analysis (useful to get an initial idea about the clustering parameters to choose) or a more statistically accurate analysis; in the second case, CIA exploits the comparison of the obtained signature scores with randomic signature scores, with the possibility to filter the scores by their distribution or their significance, and allowing the comparison of genesets with lengths spanning from tens to thousands genes.

Please read the (documentation)[] and (tutorial)[]

--------
Citation
--------
If you use ``cia`` in your work, please cite our publication as follows: 

	CIA: a Cluster Independent Annotation method to investigate cell identities in scRNA-seq data
	
	Ivan Ferrari, Mattia Battistella, Francesca Vincenti, Andrea Gobbini, Samuele Notarbartolo, 
	Jole Costanza, Stefano Biffo, Renata Grifantini, Sergio Abrignani, Eugenia Galeota
	
	bioRxiv 2023.11.30.569382; doi: (10.1101/2023.11.30.569382)[https://doi.org/10.1101/2023.11.30.569382]


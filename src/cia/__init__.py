import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "CIA"
    __version__ = version(dist_name)
    
except PackageNotFoundError:  # pragma: no cover
    __version__ = ""
finally:
    del version, PackageNotFoundError

# add to solve versioning issues
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "v1.0.0a3"

    
import seaborn as sns
import numpy as np
import pandas as pd
from anndata import AnnData
import time 
from concurrent.futures import ThreadPoolExecutor, as_completed

from .investigate import *
from .report import *
from .external import *
from .utils import *

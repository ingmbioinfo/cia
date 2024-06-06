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
    __version__ = "1.0.0a1"
finally:
    del version, PackageNotFoundError
    
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

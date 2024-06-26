"""
    Setup file for CIA.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 4.2.1.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
from setuptools import setup, find_packages

if __name__ == "__main__":
    try:
        #setup(use_scm_version={"version_scheme": "no-guess-dev"})
        #setup(use_scm_version=True, setup_requires=['setuptools_scm'])#{"version_scheme": "post-release", "local_scheme": "node-and-timestamp"})
        setup(name='cia_python',
              version='1.0.a3',  # Imposta manualmente la versione qui
              packages=find_packages(),
              install_requires=['seaborn', 'numpy', 'pandas', 'AnnData','scanpy']
        )
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "setuptools_scm and wheel with:\n"
            "   pip install -U setuptools setuptools_scm wheel\n\n"
        )
        raise

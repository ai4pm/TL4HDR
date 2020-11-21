# Deep transfer learning for reducing health care disparities arising from biomedical data inequality

This is the software package for implementing the methods and reproducing the results described in the paper [1]. The main functions of the software package include:

- Train a deep neural network to predict the clinical endpoint outcomes from multi-omics data.
- Compare the performance of multiethnic machine learning schemes.
- Use deep transfer learning to improve machine learning model performance on data-disadvantaged racial groups.
- Use simulation modeling to study the key factors influencing the performance of the multiethnic machine learning schemes.

[1] Yan Gao and Yan Cui (2020) Deep transfer learning for reducing health care disparities arising from biomedical data inequality. Nature Communications 11, 5131. https://www.nature.com/articles/s41467-020-18918-3 (Supplementary Table 1 https://github.com/ai4pm/TL4HDR/blob/master/Supplementary_Table_1.xlsx)

# Getting Started

The example folder contains the scripts for reproducing the result in Fig 3 and supplementary Fig 4.

The data folder contains the script file to ensemble datasets for the 225 machine learning tasks, 224 datasets from TCGA (https://portal.gdc.cancer.gov) and one dataset assembled from MMRF CoMMpass (https://themmrf.org/we-are-curing-multiple-myeloma/mmrf-commpass-study). Note, the mRNA expression data of TCGA cohort can be downloaded from https://figshare.com/articles/dataset/Gao_Y_Cui_Y_2020_/12811574, and put it under the TL4RDH/data/datasets/

The model folder contains the files for deep neural network and deep transfer learning implementation.

The simulation folder contains two synthetic datasets simulated using ssizeRNA (a Bioconductor package), a data sampler, and four files, for reproducing the results in Fig 4.

| **Entity** | **Path/location** | **Note** |
| --- | --- | --- |
| Deep neural network | ./model/mlp.py | The deep network model |
| Logistic regression | ./model/LogisticRegression.py | The logistic regression layer |
| Stacked auto-encoder | ./model/SdA.py | Functions to layerwise train a stacked de-noising auto-encoder. |
| Feature selection | ./ model/mlp.py/selectKBest | Feature selection for training and testing datasets. |
| Synthetic datasets 1, 3 | ./simulation/ PanGyn-DFI-5.mat | The simulation dataset using parameters estimated from PanGyn-AA/EA-Protein-DFI-5YR. |
| Synthetic datasets 2, 4 | ./simulation/ PanGyn-DFI-5-base.mat | The simulation dataset with no distribution difference. |
| Fine-tuning 1 | ./examples/classify\_util.py/ run\_supervised\_transfer\_cv | The supervised transfer learning method |
| Fine-tuning 2 | ./examples/classify\_util.py/ run\_unsupervised\_transfer\_cv | The unsupervised transfer learning, stacked auto-encoder. |
| Domain Adaptation | ./examples/classify\_util.py/ run\_CCSA\_transfer./model/CCSA/Initialization.py | The Contrastive Classification Semantic Alignment transfer learning method. |

# Installation

## Prerequisites

| **REAGENT or RESOURCE** | **SOURCE** | **IDENTIFIER** |
| --- | --- | --- |
| TCGA | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| MMRF | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/projects/MMRF-COMMPASS](https://portal.gdc.cancer.gov/projects/MMRF-COMMPASS) |
| TCGA Cancer Types | Broad Institute | [https://gdac.broadinstitute.org/](https://gdac.broadinstitute.org/) |
| American Cancer Types | Cancer Treatment Centers of America | [https://www.cancercenter.com/cancer-types](https://www.cancercenter.com/cancer-types) |
| TCGA Ancestry | TCGAA | [http://52.25.87.215/TCGAA/index.php](http://52.25.87.215/TCGAA/index.php) |
| TCGA Protein | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| TCGA mRNA | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| TCGA Clinical | Genomic Data Commons data portal | [https://portal.gdc.cancer.gov/](https://portal.gdc.cancer.gov/) |
| MMRF tool | Genomic Data Commons data portal | [https://github.com/cpreid2/gdc-rnaseq-tool](https://github.com/cpreid2/gdc-rnaseq-tool) |
| TCGA Clinical Endpoints | TCGA Pan-Cancer Clinical Data Resource | [https://www.sciencedirect.com/science/article/pii/S0092867418302290](https://www.sciencedirect.com/science/article/pii/S0092867418302290) |

| **Software and Hardware** | **SOURCE** | **IDENTIFIER** |
| --- | --- | --- |
| REAGENT or RESOURCE | SOURCE | IDENTIFIER |
| Python 2.7 | Python Software Foundation | [https://www.python.org/download/releases/2.7/](https://www.python.org/download/releases/2.7/) |
| Computational Facility | The National Institute for Computational Sciences | [https://www.nics.tennessee.edu/computing-resources/acf](https://www.nics.tennessee.edu/computing-resources/acf) |
| Numpy 1.15.4 | Tidelift, Inc | https://libraries.io/pypi/numpy/1.15.4 |
| Numpydoc 0.9.1 | Tidelift, Inc | [https://libraries.io/pypi/numpydoc](https://libraries.io/pypi/numpydoc) |
| Scipy 1.2.1 | The SciPy community | [https://docs.scipy.org/doc/scipy-1.2.1/reference/](https://docs.scipy.org/doc/scipy-1.2.1/reference/) |
| Seaborn 0.9.0 | Michael Waskom | [https://seaborn.pydata.org/installing.html](https://seaborn.pydata.org/installing.html) |
| Sklearn 0.0 | The Python community | [https://pypi.org/project/sklearn/](https://pypi.org/project/sklearn/) |
| Skrebate 0.6 | Tidelift, Inc | [https://libraries.io/pypi/skrebate](https://libraries.io/pypi/skrebate) |
| Theano 1.0.3 | LISA lab | [http://deeplearning.net/software/theano/install.html](http://deeplearning.net/software/theano/install.html) |
| Keras 2.2.4 | GitHub, Inc. | [https://github.com/keras-team/keras/releases/tag/2.2.4](https://github.com/keras-team/keras/releases/tag/2.2.4) |
| Keras-Applications 1.0.8 | GitHub, Inc. | [https://github.com/keras-team/keras-applications](https://github.com/keras-team/keras-applications) |
| Keras-Preprocessing 1.1.0 | GitHub, Inc. | [https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0](https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0) |
| Tensorboard 1.13.1 | GitHub, Inc. | [https://github.com/tensorflow/tensorboard/releases/tag/1.13.1](https://github.com/tensorflow/tensorboard/releases/tag/1.13.1) |
| Tensorflow 1.13.1 | tensorflow.org | [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) |
| Tensorflow-estimator 1.13.1 | The Python community | [https://pypi.org/project/tensorflow-estimator/](https://pypi.org/project/tensorflow-estimator/) |
| Statsmodels 0.9.0 | Statsmodels.org | [https://www.statsmodels.org/stable/release/version0.9.html](https://www.statsmodels.org/stable/release/version0.9.html) |
| Lifelines 0.16.3 | Cam Davidson-Pilon Revision | [https://lifelines.readthedocs.io/en/latest/Changelog.html](https://lifelines.readthedocs.io/en/latest/Changelog.html) |
| Optunity 1.1.1 | The Python community | [https://pypi.org/project/Optunity/](https://pypi.org/project/Optunity/) |
| Xlrd 1.2.0 | The Python community | [https://pypi.org/project/xlrd/](https://pypi.org/project/xlrd/) |
| XlsxWriter 1.1.8 | The Python community | [https://pypi.org/project/XlsxWriter/](https://pypi.org/project/XlsxWriter/) |
| Xlwings 0.15.8 | The Python community | [https://pypi.org/project/xlwings/](https://pypi.org/project/xlwings/) |
| Xlwt 1.3.0 | The Python community | [https://pypi.org/project/xlwt/](https://pypi.org/project/xlwt/) |
| Lasagne 0.2.dev1 | GitHub, Inc. | [https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne) |
|   |   |   |
| Software and Hardware |   |   |
| REAGENT or RESOURCE | SOURCE | IDENTIFIER |
| Python 2.7 | Python Software Foundation | [https://www.python.org/download/releases/2.7/](https://www.python.org/download/releases/2.7/) |
| Computational Facility | The National Institute for Computational Sciences | [https://www.nics.tennessee.edu/computing-resources/acf](https://www.nics.tennessee.edu/computing-resources/acf) |
| Numpy 1.15.4 | Tidelift, Inc | https://libraries.io/pypi/numpy/1.15.4 |
| Numpydoc 0.9.1 | Tidelift, Inc | [https://libraries.io/pypi/numpydoc](https://libraries.io/pypi/numpydoc) |
| Scipy 1.2.1 | The SciPy community | [https://docs.scipy.org/doc/scipy-1.2.1/reference/](https://docs.scipy.org/doc/scipy-1.2.1/reference/) |
| Seaborn 0.9.0 | Michael Waskom | [https://seaborn.pydata.org/installing.html](https://seaborn.pydata.org/installing.html) |
| Sklearn 0.20.2 | The Python community | [https://pypi.org/project/sklearn/](https://pypi.org/project/sklearn/) |
| Skrebate 0.6 | Tidelift, Inc | [https://libraries.io/pypi/skrebate](https://libraries.io/pypi/skrebate) |
| Theano 1.0.3 | LISA lab | [http://deeplearning.net/software/theano/install.html](http://deeplearning.net/software/theano/install.html) |
| Keras 2.2.4 | GitHub, Inc. | [https://github.com/keras-team/keras/releases/tag/2.2.4](https://github.com/keras-team/keras/releases/tag/2.2.4) |
| Keras-Applications 1.0.8 | GitHub, Inc. | [https://github.com/keras-team/keras-applications](https://github.com/keras-team/keras-applications) |
| Keras-Preprocessing 1.1.0 | GitHub, Inc. | [https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0](https://github.com/keras-team/keras-preprocessing/releases/tag/1.1.0) |
| Tensorboard 1.13.1 | GitHub, Inc. | [https://github.com/tensorflow/tensorboard/releases/tag/1.13.1](https://github.com/tensorflow/tensorboard/releases/tag/1.13.1) |
| Tensorflow 1.13.1 | tensorflow.org | [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) |
| Tensorflow-estimator 1.13.1 | The Python community | [https://pypi.org/project/tensorflow-estimator/](https://pypi.org/project/tensorflow-estimator/) |
| Statsmodels 0.9.0 | Statsmodels.org | [https://www.statsmodels.org/stable/release/version0.9.html](https://www.statsmodels.org/stable/release/version0.9.html) |
| Lifelines 0.16.3 | Cam Davidson-Pilon Revision | [https://lifelines.readthedocs.io/en/latest/Changelog.html](https://lifelines.readthedocs.io/en/latest/Changelog.html) |
| Optunity 1.1.1 | The Python community | [https://pypi.org/project/Optunity/](https://pypi.org/project/Optunity/) |
| Xlrd 1.2.0 | The Python community | [https://pypi.org/project/xlrd/](https://pypi.org/project/xlrd/) |
| XlsxWriter 1.1.8 | The Python community | [https://pypi.org/project/XlsxWriter/](https://pypi.org/project/XlsxWriter/) |
| Xlwings 0.15.8 | The Python community | [https://pypi.org/project/xlwings/](https://pypi.org/project/xlwings/) |
| Xlwt 1.3.0 | The Python community | [https://pypi.org/project/xlwt/](https://pypi.org/project/xlwt/) |
| Lasagne 0.2.dev1 | GitHub, Inc. | [https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne) |

## Running the tests

The easiest way to run the test file is bash run.sh. Download the datasets from the following link and unzip all the mat files to '/data/datasets/' folder. After the execution, the result will be saved under the &quot;Result&quot; folder. 

## References

[1] Yan Gao and Yan Cui (2020) Deep transfer learning for reducing health care disparities arising from biomedical data inequality. Nature Communications 11, 5131. https://www.nature.com/articles/s41467-020-18918-3 (Supplementary Table 1 https://github.com/ai4pm/TL4HDR/blob/master/Supplementary_Table_1.xlsx)

## Authors

Yan Gao, Yan Cui. {ygao45, ycui2}@uthsc.edu

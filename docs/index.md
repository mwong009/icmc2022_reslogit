# ICMC 2022: An application of the ResLogit model in model averaging

## Overview

### Introduction
Recently, deep learning has been successfully used in discrete choice modelling to 
improve model accuracy and performance, while at the same time, allowing for typical 
econometric analysis to be performed (Wong and Farooq, 2021)[^wongResLogit]. Unlike 
other machine learning algorithms such as decision trees and random forests, deep 
learning models such as Residual Networks. Such models do not suffer from the effects 
of “losing interpretability”. This report summarizes an implementation of the deep 
learning based ResLogit choice model (Wong and Farooq, 2021) on the three datasets.

### Model description
The model that will be used in all 3 cases is a simple 1-layer ResLogit model with 16 
hidden units. We apply the same ResLogit structure to all three datasets. We refer 
readers to the paper of Wong and Farooq (2021) for in-depth description of the model. 
In all three datasets, the utility functions for each model are kept simple. Details 
of each utility function is described in the Results section.

### Implementation
The implementation of the model is straightforward, using the python library 
[PyCMTensor](https://github.com/mwong009/pycmtensor). 

`PyCMTensor` is a discrete choice model development platform which is designed with the 
use of deep learning in mind, enabling users to write more complex models using neural 
networks. PyCMTensor aims to provide developers and researchers with deep learning 
tools for econometric and travel behaviour modelling with reproducible and 
interpretable results.

### Obtaining behavioural insights
As with conventional choice modelling estimation software such as Biogeme, we can 
obtain econometric information directly from the model output, including, but not 
limited to: (robust) Correlation matrices, statistical information (Beta values, 
t-statistics, p-value, standard errors), elasticities, WTP and so on.

Examples of obtaining elasticities are shown in the results section.

[^wongResLogit]: Wong, M. and Farooq, B., 2020. A bi-partite generative model framework for analyzing and simulating large scale multiple discrete-continuous travel behaviour data. Transportation Research Part C: Emerging Technologies, 110, pp.247-268.

## Contents

```{toctree}
:maxdepth: 3

results/index

```
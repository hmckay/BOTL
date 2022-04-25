# BOTL
This git repo has data, data generators and code for the Bi-directional Online Transfer Learning framework (BOTL).

## Running code
To run code either use
  - `python3 controller2.py ....`
  - `python3 runResults.py ....` (this runs controller2.py for multiple repeat iterations)

Use `python3 runResults.py --help` to display run options

## Common Options
  - `--domain` type of dataset used: `Following, Heating, Sudden, Gradual`
  - `--type` concept drift detection strategy: `RePro`, `ADWIN` and `AWPro` have been implemented
  - `--window` window size of the concept drift detector
  - `--runid` used when debugging (change output in `controller2.py` to somethign other than dev/null)
  - `--numStreams` number of domains in the framework
  - `--ensemble` version of BOTL/how models are combined - see below for options
  - `--perfCull` predictive performance culling threhold parameter used by P-Thresh (BOTL-C.I), MI-Thresh (BOTL-C.II) and CS-Thresh
  - `--miCull` Mutual Information culling threshold parameter used by MI-Thresh (BOTL-C.II)
  - `--paCull` Pricipal Angle/conceptual similarity culling threshold parameter used by CS-Thresh
  - `--variance` total variance captured by the PCs used to represent base models, used by CS-Thresh and CS-Clust
  - `--learner` list of types of models to be used, so far SVRs and RRs can be used
  - `--replacement` is model replacement it to be used when determining whether to transfer

## BOTL variants
Different variants of BOTL have been implemented and are specified by the `--ensemble` parameter
  - BOTL: 
    - `--ensemble OLS` 
    - BOTL with no base model selection
    - introduced in [[1]](#1), used in [[2]](#2) and [[3]](#3)
  - P-Thresh:
    - `--ensemble OLSFE`
    - use `--perfCull` to set predictive performance culling threshold parameter
    - BOTL with predictive performance thresholding to select base models
    - introduced in [[1]](#1) as BOTL-C.I, used in [[2]](#2) and [[3]](#3)
  - MI-Thresh:
    - `--ensemble OLSFEMI`
    - use `--miCull` to set mutual information culling threshold parameter
    - BOTL with mutual information and predictive performance thresholding to select base models
    - introduced in [[1]](#1) as BOTL-C.II, used in [[2]](#2) and [[3]](#3)
  - CS-Thresh:
    - `--ensemble OLSFEPA`
    - use `--paCull` to set conceptual similarity culling threshold parameter
    - BOTL with conceptual similarity and predictive performance thresholding to select base models
    - introduced in [[3]](#3)
  - CS-Clust:
    - `--ensemble OLSKPAC2`
    - use `--variance` to determine how much variance is captured within the PCs that represent the uderlying concept
    - BOTL with parameterless conceptual clustering to select base models
    - introduced in [[3]](#3), uses STSC [[4]](#4) to create clusters of similar models


## Available data and data generators and BOTL implementations:
  - Following distance data for 6 journeys (2 drivers).
  - Drifting hyperplane data generator
  - Smart home heating simulation (with real world weather data)

Note the underlying framework is the same for all three implementations. For ease of reproducibility all three versions have been added.

## AWPro
AWPro is a concept drift detection algorithm that combines aspects of RePro [[5]](#5) and ADWIN [[6]](#6) that better suit the BOTL framework. AWPro was first introduced in [[2]](#2).

## Parameter Analysis
Parameter analysis has been done to consider (see `parameterAnalysis.pdf`) to impact of the parameter values of underlying concept drift detection strategies, and how they impact the BOTL framework. 



# Source Code
The BOTL framework has been created using various code from other sources. ADWIN and AWPro implementations (which uses ADWIN as a basis for drift detection) are based upon the implementation available: https://github.com/rsdevigo/pyAdwin. This code is included in `datasetBOTL/BiDirTansfer/pyadwin/`

Other work relating to future variations of BOTL use Self-Tuning Spectral Clustering has been created based on the implementation available: https://github.com/wOOL/STSC. This code is used in `datasetBOTL/BiDirTransfer/Models/stsc*.py`


# References
<a id="1">[1]</a> 
McKay, H., Griffiths, N., Taylor, P., Damoulas, T. and Xu, Z., 2019. Online Transfer Learning for Concept Drifting Data Streams. In BigMine@ KDD.

<a id="2">[2]</a>
McKay, H., Griffiths, N., Taylor, P., Damoulas, T. and Xu, Z., 2020. Bi-directional online transfer learning: a framework. Annals of Telecommunications, 75(9), pp.523-547.

<a id="3">[3]</a>
McKay, H., Griffiths, N. and Taylor, P., 2021. Conceptually Diverse Base Model Selection for Meta-Learners in Concept Drifting Data Streams. arXiv preprint arXiv:2111.14520.

<a id="4">[4]</a>
Zelnik, M.L. and Perona, P., 2015. Self-tuning spectral clustering. Advances in Neural Information Processing Systems, pp.1601-1608.

<a id="5">[5]</a>
Yang, Y., Wu, X. and Zhu, X., 2005, August. Combining proactive and reactive predictions for data streams. In Proceedings of the eleventh ACM SIGKDD international conference on Knowledge discovery in data mining (pp. 710-715).

<a id="6">[6]</a>
Bifet, A. and Gavalda, R., 2007, April. Learning from time-changing data with adaptive windowing. In Proceedings of the 2007 SIAM international conference on data mining (pp. 443-448). Society for Industrial and Applied Mathematics.

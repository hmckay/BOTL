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
  - BOTL: `--ensemble OLS` 
        - BOTL with no base model selection



## Available data and data generators and BOTL implementations:
  - Following distance data for 6 journeys (2 drivers).
  - Drifting hyperplane data generator
  - Smart home heating simulation (with real world weather data)

Note the underlying framework is the same for all three implementations. For ease of reproducibility all three versions have been added.

## AWPro
AWPro is a concept drift detection algorithm that combines aspects of ADWIN and RePro that better suit the BOTL framework.

## Parameter Analysis
Parameter analysis has been done to consider (see `parameterAnalysis.pdf`) to impact of the parameter values of underlying concept drift detection strategies, and how they impact the BOTL framework. 


# File structure
The BOTL framework is available for Hyperplane, Heating and FollowingDistance datasets (see `reproducibility.pdf` for more information of these datasets).
BOTL has been implemented using three underlying concept drift detection algorithms: RePro, ADWIN and our own drift detector, AWPro. 



* *`controller2.py`*: manages the creation of domains in the framework and is used to transfer models between domains (sources).
* *`sourceXXXX2.py`*: a source domain using the underlying CDD XXXX (i.e. RePro, ADWIN or AWPro').
* *`Models/createModel.py`*: used to create local models and make predictions without knowledge transfer. Predictions without knowledge transfer are used to detect concept drifts
* *`Models/modelMultiConceptTransfer.py`*: used to make overarching predictions by combining the locally learnt model with models transferred from other domains (more detail below)
* *`Models/modelMultiConceptTransferHistory.py`*: used by BOTL to keep track of source models and to identify when a model is considered stable (therefore can be transferred to other domains). Also used by BOTL implementation with RePro and AWPro as underlying drift detectors to keep track of historical models and concept transitions. Allows previously learnt models to be prioritised over creating new models

<!--# BOTL with Culling (BOTL-C)-->
<!--Two BOTL-C variants are included in this repository: BOTL-C.I (model culling based on performance), and BOTL-C.II (model culling based on performance and diversity). Each of these are implemented in `Models/modelMultiConceptTransfer.py`. BOTL-C.I implementations are used when the parameter `weightType = 'OLSFE'`, and BOTL-C.II implementations are used when the parameter `weightType = 'OLSFEMI'`. In order to use these two implementations, additional parameters are needed, which are set in `controller.py` as follows:-->
<!--- *Performance threshold*: this was the original culling parameter, and therefore is denoted by parameter `CThresh` in `controller.py`-->
<!--- *Mutual Information threshold*: this parameter is denoted by `MThresh` in `controller.py`-->


# Source Code
The BOTL framework has been created using various code from other sources. ADWIN and AWPro implementations (which uses ADWIN as a basis for drift detection) are based upon the implementation available: https://github.com/rsdevigo/pyAdwin. This code is included in `datasetBOTL/BiDirTansfer/pyadwin/`

Other work relating to future variations of BOTL use Self-Tuning Spectral Clustering has been created based on the implementation available: https://github.com/wOOL/STSC. This code is used in `datasetBOTL/BiDirTransfer/Models/stsc*.py`



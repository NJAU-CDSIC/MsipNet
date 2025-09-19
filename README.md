# **MsipNet** Code Repository

This is a directory for storing the **MsipNet** model code and data.**MsipNet** is a multi-scale representation learning framework for predicting protein-RNA interactions.

---

The folders in the MsipNet repository:

- **Datasets**:
  
  a. **CLIP_seq**: 42 datasets collected from the ENCODE database and the POSTAR database.
  
  b. **RBP-24**:

  &nbsp;&nbsp;&nbsp;&nbsp;i. **All_24**:All 24 datasets.
    
  &nbsp;&nbsp;&nbsp;&nbsp;ii. **Testing_set_8**:8 preprocessed testing datasets.
    
  &nbsp;&nbsp;&nbsp;&nbsp;iii. **Testing_set_8_structure**:Predicted structures obtained using RNAfold and RNAshapes.

- **MsipNet_code**:Main code file for the MsipNet model.

- **RNA-FM**:The RNA language model.

- **SOTA**:Comparative methods used in the contrast experiments:
  
  iDeepE:https://github.com/xypan1232/iDeepE
  
  DeepCLIP:https://github.com/deepclip/deepclip
  
  GraphProt:https://github.com/dmaticzka/GraphProt
  
  PrismNet:https://github.com/kuixu/PrismNet
  
  PRIESSTESS:https://github.com/kaitlin309/PRIESSTESS
  
  HDRNet:https://github.com/zhuhr213/HDRNet
  
  PIONet:https://github.com/moodyrashid/PIONet

- **Scripts**:Contains code for motif discovery, sequence-only models, and models using predicted structural information:

  a. **Motif_discovery**: Code related to discovering motifs.
  
  b. **Predicted_structure**: Code related to using predicted structural information.
  
  c. **Sequence_only**: Code related to using sequence information only.

- **Supplementary Files**:The detailed results for all the analysis in our study.

---



### **Step-by-step Running:**

## 1. Environment Installation

It is recommended to use the conda environment (python 3.10), mainly installing the following dependencies:

- [ ] ​		**pytorch(2.0.0)、pytorch-cuda(11.8)、scipy(1.10.1)、scikit-learn(1.2.2)、pandas(2.0.0)、shap(0.41.0)、numpy(1.23.5)**

See environment.yaml for details.Use the following command to install the runtime environment.

```
conda env create -f environment.yml
```

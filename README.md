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

See environment.yaml for details. Use the following command to install the runtime environment.

```
conda env create -f environment.yml
```

## 2. Datasets

Download the datasets from the following links:

-  /Datasets/CLIP_seq:

## 3. RNA-FM

Download the RNA language model from the following link:

- ​		/RNA-FM:  https://github.com/ml4bio/RNA-FM

## 4. Training and Testing

Before running the model code, make sure that the feature files extracted by RNA-FM are stored in `/MsipNet_code/FM_embedding` with the naming format filename.pt.

- Training, using the RBM15_HepG2 dataset as an example:

```
python main.py --data_file RBM15_HepG2 --train
```

Before testing, ensure that all five model parameter files from the five-fold cross-validation are present in the `/MsipNet_code/Model` directory.

- Testing:
  
```
python main.py --data_file RBM15_HepG2 --validate
```

- Training all datasets:

```
python main.py --train_all
```

## 5. Motif Discovery

The command to obtain the motifs for a dataset is as follows. Using RBM15_HepG2 as an example, run it in the `/Scripts/Motif_discovery` directory:

```
python motif.py --file_name RBM15_HepG2
```

## 6. Predicted Structure

The command for using predicted RNA secondary structure information is as follows. Using RBM15_HepG2 as an example, run it in the `/Scripts/Predicted_struction` directory with the prediction results from RNAfold:

```
python main_RNAfold.py --data_file RBM15_HepG2 --train
python main_RNAfold.py --data_file RBM15_HepG2 --validate
python main_RNAfold.py --train_all
```

## 7. Sequence Only Model

The command for training and testing the model using sequence information only is as follows. Using RBM15_HepG2 as an example, run it in the `/Scripts/Sequence_only` directory:

```
python main_seq.py --data_file RBM15_HepG2 --train
python main_seq.py --data_file RBM15_HepG2 --validate
python main_seq.py --train_all
```

## 8. Installation

Download the code:

```
git clone https://github.com/NJAU-CDSIC/MsipNet.git
```

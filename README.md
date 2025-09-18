# **MsipNet** Code Repository

This is a directory for storing the **MsipNet** model code and data.**MsipNet** is a multi-scale representation learning framework for predicting protein-RNA interactions.

---

The folders in the MsipNet repository:

-**Datasets**:
  a. **CLIP_seq**: 42 datasets collected from the ENCODE database and the POSTAR database.
  
  b. **RBP-24**:
  
    i. **All_24**:All 24 datasets.
    
    ii. **Testing_set_8**:8 preprocessed testing datasets.
    
    iii. **Testing_set_8_structure**:Predicted structures obtained using RNAfold and RNAshapes.

-**MsipNet_code**:Main code file for the MsipNet model.

-**Scripts**:Contains code for motif discovery, sequence-only models, and models using predicted structural information:

  a. **Motif_discovery**: Code related to discovering motifs.
  
  b. **Predicted_structure**: Code related to using predicted structural information.
  
  c. **Sequence_only**: Code related to using sequence information only.

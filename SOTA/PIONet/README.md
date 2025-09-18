# PIONet: a positional encoding integrated onehot feature based RNA-binding protein classification using deep neural network
This repository contains code and resources for training and evaluating machine learning models to predict RNA-binding protein (RBP) interactions using the RBP-24 datasets. The repository includes scripts for data preprocessing, model training, and transfer learning.

## Contents
### 1. Data Folder
The data/ folder contains the RBP-24 datasets, which are used for training and evaluating the models.

### 2. Saved Models
The `saved_models/` folder stores the trained models:

- `PIONet/`: A stand-alone model trained on individual datasets from RBP-24.

- `PIONet+/`: A transfer learning model that leverages the trained base_model to improve performance.

- `basemodel_301/`: A base model trained on the merged dataset.

### 3. Scripts
`datasets_merger.py`:

- Merges all datasets from the RBP-24 collection.

- Removes duplicate entries to create a clean, unified dataset.

`train_basemodel_with_merged_data.py:`

- Trains the base_model using the merged dataset.

`train_pionet.py`:

- Trains a stand-alone model (model1) for a specific dataset from the RBP-24 collection.

`train_pionet+.py`:

- Implements a transfer learning approach by fine-tuning the base_model to improve the performance of model1.


### Dependencies
Python 3.12
- Libraries: NumPy, pandas, scikit-learn, TensorFlow/PyTorch (depending on the model implementation)

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Contact
For questions or feedback, please open an issue or contact [M Rashid] at [m.rashid@griffith.edu.au].

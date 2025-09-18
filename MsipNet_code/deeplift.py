import os
import torch
import argparse
import numpy as np
from Utils.utils import *
from sklearn.model_selection import KFold
from captum.attr import DeepLift
from Utils.MsipNet import MsipNet
import matplotlib.pyplot as plt

class TailModel(torch.nn.Module):
    """
    TailModel: A wrapper module that extracts and reuses the final feature processing
    layers (LSTM, UCDC, and fully connected) from an existing MsipNet model.
    It processes the combined features and outputs the final prediction score.
    """

    def __init__(self, original_model):
        super().__init__()
        self.lstmlast = original_model.lstmlast
        self.ucdc = original_model.ucdc
        self.fclast = original_model.fclast

    def forward(self, combined):
        combined, _ = self.lstmlast(combined)
        B, N, C = combined.size()
        H, W = 2, 10
        combined = combined.reshape(B, C, H, W)
        combined = self.ucdc(combined)
        B, C, H, W = combined.size()
        combined = combined.view(B, N, C)
        combined = combined.view(combined.size(0), -1)
        return self.fclast(combined)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.backends.cudnn.enabled = False
max_length = 101

parser = argparse.ArgumentParser(description="Process RNA file name")
parser.add_argument('--file_name', type=str, required=True, help='Name of the RNA file without extension')

args = parser.parse_args()
file_name = args.file_name

name, sequences, structs, label = read_csv_with_name('../Datasets/CLIP_seq/' + file_name + '.tsv')
seqmer = seq2mer(sequences)
print("seqmer:", seqmer.shape)
structure = np.zeros((len(structs), 1, max_length))  # (N, 1, 101)
for i in range(len(structs)):
    struct = structs[i].split(',')
    ti = [float(t) for t in struct]
    ti = np.array(ti).reshape(1, -1)
    structure[i] = np.concatenate([ti], axis=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_size = 640
hidden_size = 128
output_size = 1

fm_embeddings = torch.load("./FM_embedding/" + file_name + ".pt")
print("FM:", fm_embeddings.shape)

output_dir = f"./Deeplift_result/{file_name}/"
os.makedirs(output_dir, exist_ok=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

samples = list(zip(fm_embeddings.cpu().numpy(), structure, seqmer, label, name, sequences))
for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
    print(f"Processing Fold {fold + 1}/5 for {file_name}")
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold, val_name_fold, val_sequences_fold = zip(*val_samples)
    val_emb = torch.tensor(np.array(val_emb_fold)).float().to(device)
    val_struc = torch.tensor(np.array(val_struc_fold)).float().to(device)
    val_mer = torch.tensor(np.array(val_mer_fold)).float().to(device)

    model = MsipNet(input_size, hidden_size, output_size).to(device)
    model_file = f'./Model/{file_name}_{fold + 1}.pth'
    print("model_file:",model_file)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    features = {}
    def hook_fn(module, input, output):
        features['combined'] = input[0].detach()

    handle = model.lstmlast.register_forward_hook(hook_fn)

    tail_model = TailModel(model)
    tail_model.eval()

    with torch.no_grad():
        output = model(val_emb, val_struc, val_mer)

    combined_feature = features.get('combined', None)
    if combined_feature is None:
        print("No combined features were captured!")
        continue

    baseline = torch.zeros_like(combined_feature)
    deeplift = DeepLift(tail_model)
    attributions = deeplift.attribute(combined_feature, baseline)
    print("deeplift:",attributions.shape)
    handle.remove()

    np.save(os.path.join(output_dir, f"attributions_fold_{fold + 1}.npy"), attributions.detach().cpu().numpy())
    print(f"Fold {fold + 1} DeepLIFT analysis complete.")

num_folds = 5
fold_data_list = []

# Loop through each fold to load DeepLIFT attribution results
for fold in range(1, num_folds + 1):
    file_path = f'./Deeplift_result/{file_name}/attributions_fold_{fold}.npy'
    if os.path.exists(file_path):
        data = np.load(file_path)  # shape: (15000, 20, 384)
        fold_data_list.append(data)
        print(f"Fold {fold} loaded, shape: {data.shape}")
    else:
        print(f"File not found: {file_path}")

if not fold_data_list:
    raise ValueError("Failed to load data for any fold, please check if the path is correct.")

concat_data = np.concatenate(fold_data_list, axis=0)
mean_feature = concat_data.mean(axis=(0, 1))  # shape: (384,)

# Normalize the mean_feature values to the range [-1, 1]
min_val = mean_feature.min()
max_val = mean_feature.max()
normalized = 2 * (mean_feature - min_val) / (max_val - min_val) - 1

plt.figure(figsize=(12, 2))
plt.imshow(normalized[np.newaxis, :], cmap='bwr', aspect='auto', vmin=-1, vmax=1)
plt.colorbar()

# Draw vertical dashed lines at one-third and two-thirds of the data length
length = len(normalized)
for pos in [length // 3, 2 * length // 3]:
    plt.axvline(x=pos, color='black', linestyle='--', linewidth=1)
    plt.text(pos, 1.5, f'{pos}', ha='center', va='bottom', fontsize=8, color='black')

# Set x-axis ticks at the start, one-third, two-thirds, and end positions with corresponding labels
plt.xticks([0, length//3, 2*length//3, length-1], ['0', f'{length//3}', f'{2*length//3}', f'{length}'])
plt.yticks([])

plt.text(-5, 0, f"{file_name}", ha='center', va='center', fontsize=10, color='black', rotation=90)
plt.title(f"Attribution Heatmap (Concat of {num_folds} Folds)")

# Define the save path and save the heatmap as a 300 dpi PNG file
save_path = f"./Deeplift_result/{file_name}_concat_{num_folds}folds.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Heatmap has been saved to {save_path}")

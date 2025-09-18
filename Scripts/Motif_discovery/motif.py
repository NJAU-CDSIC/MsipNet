import os
import torch
import argparse
import numpy as np
import shap
from Utils.utils import *
import gc
from sklearn.model_selection import KFold
from Utils.MsipNet import MsipNet

parser = argparse.ArgumentParser(description="Process RNA file name")
parser.add_argument('--file_name', type=str, required=True, help='Name of the RNA file without extension')

args = parser.parse_args()
file_name = args.file_name

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False
max_length = 101
name, sequences, structs, label = read_csv_with_name('../../Datasets/CLIP_seq/' + file_name + '.tsv')

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

fm_embeddings = torch.load("../../MsipNet_code/FM_embedding/" + file_name + ".pt")
print("FM_embedding:", fm_embeddings.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kf = KFold(n_splits=5, shuffle=True, random_state=42)

samples = list(zip(fm_embeddings.cpu().numpy(), structure, seqmer, label, name, sequences))
for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
    print(f"Processing Fold {fold + 1}/5 for {file_name}")
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold, train_name_fold, train_sequences_fold = zip(
        *train_samples)

    train_emb_fold = np.array(train_emb_fold)
    train_struc_fold = np.array(train_struc_fold)
    train_mer_fold = np.array(train_mer_fold)

    fm_embeddings = train_emb_fold
    structure = train_struc_fold
    seqmer = train_mer_fold

    print("FM_embedding:", fm_embeddings.shape)
    print("structure:", structure.shape)
    print("seqmer:", seqmer.shape)

    model = MsipNet(input_size, hidden_size, output_size).to(device)

    # Load the pre-trained model from the specified file path
    model_file = '../../MsipNet_code/Model/' + file_name + '_1.pth'
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Perform prediction on the input features (fm_embeddings, structure, seqmer)
    with torch.no_grad():
        prob = torch.sigmoid(
            model(torch.tensor(fm_embeddings).to(device).float(), torch.tensor(structure).to(device).float(),
                  torch.tensor(seqmer).to(device).float()))
    prob_seleced = prob.detach().cpu().numpy()

    # Find the indices of predictions where probability exceeds the threshold
    threshold = 0.8
    selected_indices = np.where(prob_seleced > threshold)[0]
    selected_names = [train_name_fold[i] for i in selected_indices]
    selected_sequences = [train_sequences_fold[i] for i in selected_indices]
    selected_predictions = [prob_seleced[i] for i in selected_indices]

    output_file = './Results/Selected_seq/' + file_name + '.tsv'
    with open(output_file, 'w') as f:
        f.write("Name\tSequence\tPrediction\n")
        for i in range(len(selected_indices)):
            f.write(f"{selected_names[i]}\t{selected_sequences[i]}\t{selected_predictions[i][0]:.4f}\n")

    print(f"Filtered results have been saved to {output_file}")

    fm_embeddings = fm_embeddings[selected_indices]
    seqmer = seqmer[selected_indices]
    structure = structure[selected_indices]

    print("fm_embedding_selected:", fm_embeddings.shape)
    print("structure_selected:", structure.shape)
    print("seqmer_selected:", seqmer.shape)

    # SHAP explanation setup
    torch.cuda.empty_cache()
    test_emb = torch.tensor(fm_embeddings).requires_grad_().to(device).type(torch.float32)
    test_one = torch.tensor(seqmer).requires_grad_().to(device).type(torch.float32)
    test_struc = torch.tensor(structure).requires_grad_().to(device).type(torch.float32)
    e = shap.GradientExplainer(model, [test_emb, test_struc, test_one])

    mers_all = []
    scores_all = []
    for i in range(len(test_emb)):
        shap_values = e.shap_values([test_emb[i:i + 1], test_struc[i:i + 1], test_one[i:i + 1]])


        def norm(data):
            _range = np.max(data, axis=1) - np.min(data, axis=1)
            return (data - np.min(data, axis=1)) / _range


        input_seq = selected_sequences[i]
        input_struct = test_struc[i]

        # Extract attention scores from SHAP values
        fm_attention_data = np.max(shap_values[0], axis=2)  # [1, 101]
        hot = np.max(shap_values[2], axis=1)
        new_hot = np.zeros((1, 101))
        # Expand one-hot attention data to length 101 with smoothing
        for j in range(101):
            if j == 0:
                new_hot[0, j] = hot[0, j]
            elif j == 1:
                new_hot[0, j] = (hot[0, j - 1] + hot[0, j]) / 2
            elif j == 99:
                new_hot[0, j] = (hot[0, j - 1] + hot[0, j - 2]) / 2
            elif j == 100:
                new_hot[0, j] = hot[0, j - 2]
            else:
                new_hot[0, j] = (hot[0, j - 2] + hot[0, j - 1] + hot[0, j]) / 3

        hot = new_hot
        struc_attention_data = shap_values[1][0]
        # Sum attention scores across positions for feature modalities
        shap_emb = np.sum(fm_attention_data, axis=1)
        shap_one = np.sum(hot, axis=1)
        # **Compute contribution weights for each channel**
        total_contribution = np.abs(shap_emb) + np.abs(shap_one) + 1e-8  # 避免除零
        weight_emb = np.abs(shap_emb) / total_contribution
        weight_one = np.abs(shap_one) / total_contribution
        W = np.concatenate([fm_attention_data, hot, struc_attention_data], axis=0)
        # Handle missing values in the structure input
        x_str = input_struct.cpu().detach().numpy().reshape(101, 1)
        str_null = np.zeros_like(x_str)
        ind = np.where(x_str == -1)[0]
        str_null[ind, 0] = 1
        str_null = np.squeeze(str_null).T

        input_seq_array = np.array(list(input_seq)).reshape(1, -1)
        X = np.concatenate([input_seq_array, input_struct.cpu().detach().numpy()], axis=0)
        import Utils.visualize as visualize

        result = visualize.get_region_no_nor(X, W, weight_emb, weight_one)
        if result is None:
            continue

        mers, scores = result
        mers_all.extend(mers)
        scores_all.extend(scores)

    unique_mers = {}

    # Iterate through all k-mers and their scores
    for mer, score in zip(mers_all, scores_all):
        if mer in unique_mers:
            unique_mers[mer] = max(unique_mers[mer], score)
        else:
            unique_mers[mer] = score

    mers_all = list(unique_mers.keys())
    scores_all = list(unique_mers.values())
    top_kmers = [mer for _, mer in sorted(zip(scores_all, mers_all), reverse=True)[:1000]]
    output_path = "./Results/Split_seq/" + file_name + "_top_1000_mers.fasta"

    with open(output_path, "w") as f:
        for i, mer in enumerate(top_kmers):
            f.write(f">seq{i + 1}\n{mer}\n")

    print(f"Top 1000 k-mers have been saved to {output_path}")
    break

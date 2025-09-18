import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data
import torch.optim as optim

from Utils.train_loop import trainstr, validatestr
from Utils.utils import read_csv, myDataset, GradualWarmupScheduler, seq2mer
from sklearn.model_selection import KFold
from Utils.MsipNet import MsipNet

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    # print("[Info] cudnn.deterministic set to True. CUDNN-optimized code may be slow.")


def main(args):
    try:
        from termcolor import cprint
    except ImportError:
        cprint = None

    try:
        from pycrayon import CrayonClient
    except ImportError:
        CrayonClient = None

    def log_print(text, color=None, on_color=None, attrs=None):
        if cprint is not None:
            cprint(text, color=color, on_color=on_color, attrs=attrs)
        else:
            print(text)

    fix_seed(args.seed)  # fix seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_length = 101

    file_name = args.data_file
    data_path = args.data_path
    fm_path = args.fm_path
    result_save_path = args.result_save_path
    model_save_path = args.model_save_path

    if args.train:
        # Load RNA sequences, structures, and labels from a TSV file
        sequences, structs, label = read_csv(os.path.join(data_path, file_name + '.tsv'))
        # Load precomputed RNA-FM embeddings and convert to NumPy array
        embeddings = torch.load(os.path.join(fm_path, file_name + ".pt"))
        print("FM_embeddin:", embeddings.shape)
        embeddings = embeddings.cpu().detach().numpy()

        seqmer = seq2mer(sequences)
        print("mer:", seqmer.shape)

        samples = list(zip(embeddings, seqmer, label))

        # output_dir = f"{result_save_path}/{file_name}/"
        # os.makedirs(output_dir, exist_ok=True)
        # metrics_file = os.path.join(output_dir, "metrics.txt")
        # with open(metrics_file, 'w') as f:
        #     f.write(
        #         "Fold\tBest ACC\tBest PR\tBest Recall\tBest Specificity\tBest MCC\tBest F1-socre\tBest AUC\tBest AP\n")

        # Set up 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        best_overall_model = None
        best_overall_auc = 0

        for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
            if fold+1==5:
                print(f"Fold {fold + 1}/5")

                # Split samples into training and validation sets for the current fold
                train_samples = [samples[i] for i in train_idx]
                val_samples = [samples[i] for i in val_idx]

                train_emb_fold, train_mer_fold, train_label_fold = zip(*train_samples)
                val_emb_fold, val_mer_fold, val_label_fold = zip(*val_samples)

                # Separate features and labels and convert to NumPy arrays
                train_emb_fold = np.array(train_emb_fold)
                train_mer_fold = np.array(train_mer_fold)
                train_label_fold = np.array(train_label_fold)

                val_emb_fold = np.array(val_emb_fold)
                val_mer_fold = np.array(val_mer_fold)
                val_label_fold = np.array(val_label_fold)

                train_set = myDataset(train_emb_fold, train_mer_fold, train_label_fold)
                val_set = myDataset(val_emb_fold, val_mer_fold, val_label_fold)

                train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_set, batch_size=32 * 8, shuffle=False)

                input_size = 640
                hidden_size = 128
                output_size = 1

                # Initialize tracking variables for best metrics and early stopping
                best_auc = 0
                best_acc = 0
                best_prc = 0
                best_epoch = 0
                best_F1_score = 0
                best_precision = 0
                best_recall = 0
                best_mcc = 0
                best_specificity = 0
                early_stopping = 20

                # Initialize model, loss function, optimizer, and learning rate scheduler
                model = MsipNet(input_size, hidden_size, output_size).to(device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
                optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-6)
                scheduler = GradualWarmupScheduler(
                    optimizer, multiplier=8, total_epoch=float(200), after_scheduler=None)

                for epoch in range(1, 200):
                    model.train()
                    t_met = trainstr(model, device, train_loader, criterion, optimizer, batch_size=32)
                    v_met, true_labels, predictions = validatestr(model, device, val_loader, criterion)
                    scheduler.step()
                    lr = scheduler.get_lr()[0]
                    color_best = 'green'

                    if best_auc < v_met.auc:
                        best_auc = v_met.auc
                        best_acc = v_met.acc
                        best_prc = v_met.prc
                        best_F1_score = v_met.f1_score
                        best_precision = v_met.precision
                        best_recall = v_met.recall
                        best_mcc = v_met.mcc
                        best_specificity = v_met.specificity
                        best_epoch = epoch
                        color_best = 'red'
                        # with open(os.path.join(output_dir, f"fold_{fold + 1}.txt"), 'w') as pred_file:
                        #     for p, t in zip(predictions, true_labels):
                        #         pred_file.write(f"{p}\t{t}\n")
                    if epoch - best_epoch > early_stopping:
                        print("Early stop at %d, %s " % (epoch, 'MsipNet'))
                        break
                    line = '{} \t Train Epoch: {}     avg.loss: {:.4f} ACC: {:.2f}%, PR: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, MCC: {:.4f} ,F1-socre: {:.4f}, AUC: {:.4f}, AP: {:.4f} lr: {:.6f}'.format(
                        file_name, epoch, t_met.other[0], t_met.acc, t_met.precision, t_met.recall, t_met.specificity,
                        t_met.mcc,
                        t_met.f1_score, t_met.auc, t_met.prc, lr)  # 打印文件名称、轮次、损失率平均数、准确率、AUC、学习率
                    log_print(line, color='green', attrs=['bold'])

                    line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} ACC: {:.2f}%, PR: {:.4f} , Recall: {:.4f} ,Specificity: {:.4f}, MCC: {:.4f}, F1-socre: {:.4f}, AUC: {:.4f} ({:.4f}), AP: {:.4f} {}'.format(
                        file_name, epoch, v_met.other[0], v_met.acc, v_met.precision, v_met.recall, v_met.specificity,
                        v_met.mcc, v_met.f1_score, v_met.auc, best_auc, v_met.prc, best_epoch)
                    log_print(line, color=color_best, attrs=['bold'])

                # with open(metrics_file, 'a') as f:
                #     f.write(
                #         f"{fold + 1}\t{best_acc:.4f}\t{best_precision:.4f}\t{best_recall:.4f}\t{best_specificity:.4f}\t{best_mcc:.4f}\t{best_F1_score:.4f}\t{best_auc:.4f}\t{best_prc:.4f}\n")

                print(
                    "{} ACC: {:.4f} PR: {:.4f} Recall:{:.4f} Specificity: {:.4f} MCC: {:.4f} F1-socre: {:.4f} AUC: {:.4f} AP: {:.4f}".format(
                        file_name, best_acc, best_precision, best_recall, best_specificity, best_mcc, best_F1_score,
                        best_auc,
                        best_prc))

                if best_auc > best_overall_auc:
                    best_overall_auc = best_auc
                    best_overall_model = model.state_dict()

                    print("=================Saving model")
                    torch.save(best_overall_model,
                               f"{model_save_path}/" + file_name + "_" + str(fold + 1) + ".pth")
                    print("Done!")
            else:
                print(f"Skipping Fold {fold + 1}/5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to MsipNet!')
    parser.add_argument('--data_file', default='RBM15_HepG2', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='../../Datasets/CLIP_seq', type=str, help='The data path')
    parser.add_argument('--fm_path', default='../../MsipNet_code/FM_embedding', type=str,
                        help='FM embedding path')
    parser.add_argument('--model_save_path', default='./Model', type=str,
                        help='Save the trained model')
    parser.add_argument('--result_save_path', default='./Results', type=str,
                        help='Save the model result')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--train_all', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=20)

    args = parser.parse_args()
    main(args)
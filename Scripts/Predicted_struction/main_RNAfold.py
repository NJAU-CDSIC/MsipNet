import os
import gc
import glob
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data
import torch.optim as optim

from Utils.train_loop import train, validate
from Utils.utils import read_csv, myDataset, GradualWarmupScheduler, seq2mer, RNAfold_to_onehot, read_fasta_with_RNAfold
from sklearn.model_selection import KFold
from Utils.MsipNet_RNAfold import MsipNet

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
        sequences, structs, label = read_csv(os.path.join(data_path, file_name+'.tsv'))
        # Load precomputed RNA-FM embeddings and convert to NumPy array
        embeddings = torch.load(os.path.join(fm_path, file_name + ".pt"))
        print("FM_embeddin:", embeddings.shape)
        embeddings = embeddings.cpu().detach().numpy()

        seqmer = seq2mer(sequences)
        print("mer:", seqmer.shape)

        # Convert structure strings to numerical arrays with shape (N, 1, 101)
        # structure = np.zeros((len(structs), 1, 101))  # (N, 1, 101)
        # for i in range(len(structs)):
        #     struct = structs[i].split(',')
        #     ti = [float(t) for t in struct]
        #     ti = np.array(ti).reshape(1, -1)
        #     structure[i] = np.concatenate([ti], axis=0)

        _, structures, _ = read_fasta_with_RNAfold("./RNAfold/" + file_name + ".txt")
        structure = RNAfold_to_onehot(structures)
        print("RNAfold:",structure.shape)

        samples = list(zip(embeddings, structure, seqmer, label))

        output_dir = f"{result_save_path}/{file_name}/"
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, "metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(
                "Fold\tBest ACC\tBest PR\tBest Recall\tBest Specificity\tBest MCC\tBest F1-socre\tBest AUC\tBest AP\n")

        # Set up 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        auc_list = []
        acc_list = []
        prc_list = []
        f1_score_list = []
        precision_list = []
        recall_list = []
        mcc_list = []
        specificity_list = []

        best_overall_model = None
        best_overall_auc = 0

        for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
            print(f"Fold {fold + 1}/5")

            # Split samples into training and validation sets for the current fold
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold = zip(*train_samples)
            val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold = zip(*val_samples)

            # Separate features and labels and convert to NumPy arrays
            train_emb_fold = np.array(train_emb_fold)
            train_struc_fold = np.array(train_struc_fold)
            train_mer_fold = np.array(train_mer_fold)
            train_label_fold = np.array(train_label_fold)

            val_emb_fold = np.array(val_emb_fold)
            val_struc_fold = np.array(val_struc_fold)
            val_mer_fold = np.array(val_mer_fold)
            val_label_fold = np.array(val_label_fold)

            train_set = myDataset(train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold)
            val_set = myDataset(val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold)

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
                t_met = train(model, device, train_loader, criterion, optimizer, batch_size=32)
                v_met, true_labels, predictions = validate(model, device, val_loader, criterion)
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
                    with open(os.path.join(output_dir, f"fold_{fold + 1}.txt"), 'w') as pred_file:
                        for p, t in zip(predictions, true_labels):
                            pred_file.write(f"{p}\t{t}\n")
                if epoch - best_epoch > early_stopping:
                    print("Early stop at %d, %s " % (epoch, 'MsipNet'))
                    break
                line = '{} \t Train Epoch: {}     avg.loss: {:.4f} ACC: {:.2f}%, PR: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, MCC: {:.4f} ,F1-socre: {:.4f}, AUC: {:.4f}, AP: {:.4f} lr: {:.6f}'.format(
                    file_name, epoch, t_met.other[0], t_met.acc, t_met.precision, t_met.recall, t_met.specificity, t_met.mcc,
                    t_met.f1_score, t_met.auc, t_met.prc, lr)  # 打印文件名称、轮次、损失率平均数、准确率、AUC、学习率
                log_print(line, color='green', attrs=['bold'])

                line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} ACC: {:.2f}%, PR: {:.4f} , Recall: {:.4f} ,Specificity: {:.4f}, MCC: {:.4f}, F1-socre: {:.4f}, AUC: {:.4f} ({:.4f}), AP: {:.4f} {}'.format(
                    file_name, epoch, v_met.other[0], v_met.acc, v_met.precision, v_met.recall, v_met.specificity,
                    v_met.mcc, v_met.f1_score, v_met.auc, best_auc,v_met.prc, best_epoch)
                log_print(line, color=color_best, attrs=['bold'])

            with open(metrics_file, 'a') as f:
                f.write(
                    f"{fold + 1}\t{best_acc:.4f}\t{best_precision:.4f}\t{best_recall:.4f}\t{best_specificity:.4f}\t{best_mcc:.4f}\t{best_F1_score:.4f}\t{best_auc:.4f}\t{best_prc:.4f}\n")

            print(
                "{} ACC: {:.4f} PR: {:.4f} Recall:{:.4f} Specificity: {:.4f} MCC: {:.4f} F1-socre: {:.4f} AUC: {:.4f} AP: {:.4f}".format(
                    file_name, best_acc, best_precision, best_recall, best_specificity, best_mcc, best_F1_score, best_auc,
                    best_prc))

            if best_auc > best_overall_auc:
                best_overall_auc = best_auc
                best_overall_model = model.state_dict()

                print("=================Saving model")
                torch.save(best_overall_model,
                           f"{model_save_path}/" + file_name + "_" + str(fold + 1) + ".pth")
                print("Done!")

            auc_list.append(best_auc)
            acc_list.append(best_acc)
            prc_list.append(best_prc)
            f1_score_list.append(best_F1_score)
            precision_list.append(best_precision)
            recall_list.append(best_recall)
            mcc_list.append(best_mcc)
            specificity_list.append(best_specificity)

        # Compute the average performance metrics across all cross-validation folds
        avg_auc = np.mean(auc_list)
        avg_acc = np.mean(acc_list)
        avg_prc = np.mean(prc_list)
        avg_f1_score = np.mean(f1_score_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_mcc = np.mean(mcc_list)
        avg_specificity = np.mean(specificity_list)

        print(f"Average ACC: {avg_acc:.4f}")
        print(f"Average PR: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average Specificity: {avg_specificity:.4f}")
        print(f"Average MCC: {avg_mcc:.4f}")
        print(f"Average F1 Score: {avg_f1_score:.4f}")
        print(f"Average AUC: {avg_auc:.4f}")
        print(f"Average AP: {avg_prc:.4f}")

    if args.validate:  # validate only. WARNING: PLEASE FIX SEED BEFORE VALIDATION.
        # Load RNA sequences, structures, and labels from a TSV file
        sequences, structs, label = read_csv(os.path.join(data_path, file_name+'.tsv'))
        # Load precomputed RNA-FM embeddings and convert to NumPy array
        embeddings = torch.load(os.path.join(fm_path, file_name + ".pt"))
        print("FM_embeddin:", embeddings.shape)
        embeddings = embeddings.cpu().detach().numpy()

        seqmer = seq2mer(sequences)
        print("mer:", seqmer.shape)

        # Convert structure strings to numerical arrays with shape (N, 1, 101)
        # structure = np.zeros((len(structs), 1, 101))  # (N, 1, 101)
        # for i in range(len(structs)):
        #     struct = structs[i].split(',')
        #     ti = [float(t) for t in struct]
        #     ti = np.array(ti).reshape(1, -1)
        #     structure[i] = np.concatenate([ti], axis=0)

        _, structures, _ = read_fasta_with_RNAfold("./RNAfold/" + file_name + ".txt")
        structure = RNAfold_to_onehot(structures)
        print("RNAfold:",structure.shape)

        samples = list(zip(embeddings, structure, seqmer, label))

        # Initialize 5-fold cross-validation with shuffling and fixed random seed
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        auc_list = []
        acc_list = []
        prc_list = []
        f1_score_list = []
        precision_list = []
        recall_list = []
        mcc_list = []
        specificity_list = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
            print(f"Fold {fold + 1}/5")

            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold = zip(*train_samples)
            val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold = zip(*val_samples)

            train_emb_fold = np.array(train_emb_fold)
            train_struc_fold = np.array(train_struc_fold)
            train_mer_fold = np.array(train_mer_fold)
            train_label_fold = np.array(train_label_fold)

            val_emb_fold = np.array(val_emb_fold)
            val_struc_fold = np.array(val_struc_fold)
            val_mer_fold = np.array(val_mer_fold)
            val_label_fold = np.array(val_label_fold)

            train_set = myDataset(train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold)
            val_set = myDataset(val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold)

            train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=32 * 8, shuffle=False)

            input_size = 640
            hidden_size = 128
            output_size = 1

            # Initialize the MsipNet model and move it to the specified device
            model = MsipNet(input_size, hidden_size, output_size).to(device)
            # Load the trained model weights for the current fold
            model_file = f'{model_save_path}/' + file_name + f'_{fold+1}.pth'
            model.load_state_dict(torch.load(model_file))
            model.eval()
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
            v_met, true_labels, predictions = validate(model, device, val_loader, criterion)

            auc_list.append(v_met.auc)
            acc_list.append(v_met.acc)
            prc_list.append(v_met.prc)
            f1_score_list.append(v_met.f1_score)
            precision_list.append(v_met.precision)
            recall_list.append(v_met.recall)
            mcc_list.append(v_met.mcc)
            specificity_list.append(v_met.specificity)

        avg_auc = np.mean(auc_list)
        avg_acc = np.mean(acc_list)
        avg_prc = np.mean(prc_list)
        avg_f1_score = np.mean(f1_score_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_mcc = np.mean(mcc_list)
        avg_specificity = np.mean(specificity_list)

        print(f"Average ACC: {avg_acc:.4f}")
        print(f"Average PR: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average Specificity: {avg_specificity:.4f}")
        print(f"Average MCC: {avg_mcc:.4f}")
        print(f"Average F1 Score: {avg_f1_score:.4f}")
        print(f"Average AUC: {avg_auc:.4f}")
        print(f"Average AP: {avg_prc:.4f}")

    if args.train_all:
        # Get all .tsv file names (without extensions) under the data_path directory
        tsv_files = glob.glob(os.path.join(data_path, '*.tsv'))
        file_names = [os.path.splitext(os.path.basename(f))[0] for f in tsv_files]
        for file_name in file_names:
            # Load RNA sequences, structures, and labels from a TSV file
            sequences, structs, label = read_csv(os.path.join(data_path, file_name + '.tsv'))
            # Load precomputed RNA-FM embeddings and convert to NumPy array
            embeddings = torch.load(os.path.join(fm_path, file_name + ".pt"))
            print("FM_embeddin:", embeddings.shape)
            embeddings = embeddings.cpu().detach().numpy()

            seqmer = seq2mer(sequences)
            print("mer:", seqmer.shape)

            # Convert structure strings to numerical arrays with shape (N, 1, 101)
            # structure = np.zeros((len(structs), 1, 101))  # (N, 1, 101)
            # for i in range(len(structs)):
            #     struct = structs[i].split(',')
            #     ti = [float(t) for t in struct]
            #     ti = np.array(ti).reshape(1, -1)
            #     structure[i] = np.concatenate([ti], axis=0)

            _, structures, _ = read_fasta_with_RNAfold(
                "./RNAfold/" + file_name + ".txt")
            structure = RNAfold_to_onehot(structures)
            print("RNAfold:", structure.shape)

            samples = list(zip(embeddings, structure, seqmer, label))

            output_dir = f"{result_save_path}/{file_name}/"
            os.makedirs(output_dir, exist_ok=True)
            metrics_file = os.path.join(output_dir, "metrics.txt")
            with open(metrics_file, 'w') as f:
                f.write(
                    "Fold\tBest ACC\tBest PR\tBest Recall\tBest Specificity\tBest MCC\tBest F1-socre\tBest AUC\tBest AP\n")

            # Set up 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            auc_list = []
            acc_list = []
            prc_list = []
            f1_score_list = []
            precision_list = []
            recall_list = []
            mcc_list = []
            specificity_list = []

            best_overall_model = None
            best_overall_auc = 0

            for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
                print(f"Fold {fold + 1}/5")

                # Split samples into training and validation sets for the current fold
                train_samples = [samples[i] for i in train_idx]
                val_samples = [samples[i] for i in val_idx]

                train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold = zip(*train_samples)
                val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold = zip(*val_samples)

                # Separate features and labels and convert to NumPy arrays
                train_emb_fold = np.array(train_emb_fold)
                train_struc_fold = np.array(train_struc_fold)
                train_mer_fold = np.array(train_mer_fold)
                train_label_fold = np.array(train_label_fold)

                val_emb_fold = np.array(val_emb_fold)
                val_struc_fold = np.array(val_struc_fold)
                val_mer_fold = np.array(val_mer_fold)
                val_label_fold = np.array(val_label_fold)

                train_set = myDataset(train_emb_fold, train_struc_fold, train_mer_fold, train_label_fold)
                val_set = myDataset(val_emb_fold, val_struc_fold, val_mer_fold, val_label_fold)

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
                    t_met = train(model, device, train_loader, criterion, optimizer, batch_size=32)
                    v_met, true_labels, predictions = validate(model, device, val_loader, criterion)
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
                        with open(os.path.join(output_dir, f"fold_{fold + 1}.txt"), 'w') as pred_file:
                            for p, t in zip(predictions, true_labels):
                                pred_file.write(f"{p}\t{t}\n")
                    if epoch - best_epoch > early_stopping:
                        print("Early stop at %d, %s " % (epoch, 'MsipNet'))
                        break
                    line = '{} \t Train Epoch: {}     avg.loss: {:.4f} ACC: {:.2f}%, PR: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, MCC: {:.4f} ,F1-socre: {:.4f}, AUC: {:.4f}, AP: {:.4f} lr: {:.6f}'.format(
                        file_name, epoch, t_met.other[0], t_met.acc, t_met.precision, t_met.recall,
                        t_met.specificity,
                        t_met.mcc,
                        t_met.f1_score, t_met.auc, t_met.prc, lr)  # 打印文件名称、轮次、损失率平均数、准确率、AUC、学习率
                    log_print(line, color='green', attrs=['bold'])

                    line = '{} \t Test  Epoch: {}     avg.loss: {:.4f} ACC: {:.2f}%, PR: {:.4f} , Recall: {:.4f} ,Specificity: {:.4f}, MCC: {:.4f}, F1-socre: {:.4f}, AUC: {:.4f} ({:.4f}), AP: {:.4f} {}'.format(
                        file_name, epoch, v_met.other[0], v_met.acc, v_met.precision, v_met.recall,
                        v_met.specificity,
                        v_met.mcc, v_met.f1_score, v_met.auc, best_auc, v_met.prc, best_epoch)
                    log_print(line, color=color_best, attrs=['bold'])

                with open(metrics_file, 'a') as f:
                    f.write(
                        f"{fold + 1}\t{best_acc:.4f}\t{best_precision:.4f}\t{best_recall:.4f}\t{best_specificity:.4f}\t{best_mcc:.4f}\t{best_F1_score:.4f}\t{best_auc:.4f}\t{best_prc:.4f}\n")

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

                auc_list.append(best_auc)
                acc_list.append(best_acc)
                prc_list.append(best_prc)
                f1_score_list.append(best_F1_score)
                precision_list.append(best_precision)
                recall_list.append(best_recall)
                mcc_list.append(best_mcc)
                specificity_list.append(best_specificity)

            # Compute the average performance metrics across all cross-validation folds
            avg_auc = np.mean(auc_list)
            avg_acc = np.mean(acc_list)
            avg_prc = np.mean(prc_list)
            avg_f1_score = np.mean(f1_score_list)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_mcc = np.mean(mcc_list)
            avg_specificity = np.mean(specificity_list)

            print(f"Average ACC: {avg_acc:.4f}")
            print(f"Average PR: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Average Specificity: {avg_specificity:.4f}")
            print(f"Average MCC: {avg_mcc:.4f}")
            print(f"Average F1 Score: {avg_f1_score:.4f}")
            print(f"Average AUC: {avg_auc:.4f}")
            print(f"Average AP: {avg_prc:.4f}")

            torch.cuda.empty_cache()
            gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to MsipNet!')
    parser.add_argument('--data_file', default='RBM15_HepG2', type=str, help='RBP to train or validate')
    parser.add_argument('--data_path', default='../../Datasets/CLIP_seq', type=str, help='The data path')
    parser.add_argument('--fm_path', default='../../MsipNet_code/FM_embedding', type=str,
                        help='FM embedding path')
    parser.add_argument('--model_save_path', default='./RNAfold_model', type=str,
                        help='Save the trained model')
    parser.add_argument('--result_save_path', default='./RNAfold_results', type=str,
                        help='Save the model result')

    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--validate', default=False, action='store_true')
    parser.add_argument('--train_all', default=False, action='store_true')

    parser.add_argument('--seed', default=1024, type=int, help='The random seed')
    parser.add_argument('--early_stopping', type=int, default=20)

    args = parser.parse_args()
    main(args)
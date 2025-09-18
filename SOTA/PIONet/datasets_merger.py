import numpy as np
import pandas as pd
import os
import tqdm

from sklearn.utils import shuffle
# from sklearn.model_selection import train_test_split

experiments=[
    'ALKBH5_Baltz2012',
    'C17ORF85_Baltz2012',
    'C22ORF28_Baltz2012',
    'CAPRIN1_Baltz2012',
    'CLIPSEQ_AGO2',
    'CLIPSEQ_ELAVL1',
    'CLIPSEQ_SFRS1',
    'ICLIP_HNRNPC',
    'ICLIP_TDP43',
    'ICLIP_TIA1',
    'ICLIP_TIAL1',
    'PARCLIP_AGO1234',
    'PARCLIP_ELAVL1',
    'PARCLIP_ELAVL1A',
    'PARCLIP_EWSR1',
    'PARCLIP_FUS',
    'PARCLIP_HUR',
    'PARCLIP_IGF2BP123',
    'PARCLIP_MOV10_Sievers',
    'PARCLIP_PUM2',
    'PARCLIP_QKI',
    'PARCLIP_TAF15',
    'PTBv1',
    'ZC3H7B_Baltz2012'
]

unique_pos_seq = []

saved_path = os.path.join(os.getcwd(),'data/RBP_24')
data_path=os.path.join(os.getcwd(),"data/RBP_24/GraphProt_CLIP_sequences")

def read_seq_graphprot(seq_file, label=1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)

    return seq_list, labels


poscount = 0
negcount = 0
total_pos_seq = []
total_neg_seq = []

total_train_seq = []
total_train_lab = []

total_test_seq = []
total_test_lab = []

for i in tqdm.tqdm(range(24)):
    dataset = experiments[i]
    # training sequences for both positive and negative
    # ## positive sequences
    positive_train_file = os.path.join(data_path, f'{dataset}.train.positives.fa')
    pos_seq_train, pos_lab_train = read_seq_graphprot(positive_train_file, label=1)
    total_train_seq = total_train_seq + pos_seq_train
    total_train_lab = total_train_lab + pos_lab_train

    # ## negative sequences
    negative_train_file = os.path.join(data_path, f'{dataset}.train.negatives.fa')
    neg_seq_train, neg_lab_train = read_seq_graphprot(negative_train_file, label=0)
    total_train_seq = total_train_seq + neg_seq_train
    total_train_lab = total_train_lab + neg_lab_train

    # test sequences for both positive and negative
    # ## positive sequences
    positive_test_file = os.path.join(data_path, f'{dataset}.ls.positives.fa')
    pos_seq_test, pos_lab_test=read_seq_graphprot(positive_test_file, label=1)
    total_test_seq=total_test_seq+pos_seq_test
    total_test_lab=total_test_lab+pos_lab_test

    # ## negative sequences
    negative_test_file = os.path.join(data_path, f'{dataset}.ls.negatives.fa')
    neg_seq_test, neg_lab_test=read_seq_graphprot(negative_test_file, label=0)
    total_test_seq=total_test_seq+neg_seq_test
    total_test_lab=total_test_lab+neg_lab_test

print(f"Total train X/Y: {len(total_train_seq)}/{len(total_train_lab)}")

# Train
train_data={"sequence":total_train_seq, "label":total_train_lab}
df_train=pd.DataFrame(data=train_data)
# df_train[df_train.duplicated()]
df_train=df_train.drop_duplicates() # remove duplicate sequences

# Test
test_data={"sequence":total_test_seq, "label":total_test_lab}
df_test=pd.DataFrame(data=test_data)

# Remove if any test sequence present in the train
df_train=df_train[~df_train['sequence'].isin(df_test['sequence'])]

seq_train =df_train["sequence"].to_list()
lab_train =df_train["label"].to_list()

total_train_seq, total_train_lab = shuffle(seq_train, lab_train)
train_data={"sequence":total_train_seq, "label":total_train_lab}

df_train=pd.DataFrame(data=train_data)
df_train.to_csv(os.path.join(saved_path,'all_x24_merged.csv'), index=False)


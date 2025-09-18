import numpy as np
import os
import pickle as pkl

from datetime import datetime

from sklearn.utils import shuffle
from sklearn import metrics
import tqdm
import tensorflow as tf
import keras
from keras import layers
import argparse

BASE='AUGC'

MAX_SLICES=2
OVERLAP_SIZE=50
MOTIF_SIZE=4
WINDOW_SIZE=301
IN_CHANNELS=16
#=====================================================MODEL++++++++++++++++++++++++++++++++

# Define a residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolution
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adding the shortcut connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# Define the ResNet model
def create_resnet_2D(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    filters=128
    # Initial convolution layer
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, filters=filters)

    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Flatten()(x)

    x = layers.Dense (512, activation='relu')(x)

    x = layers.Dense (num_classes, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, x)
    return model



def positional_encoding(seq_length, embedding_dim):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))
    pos_enc = np.zeros((seq_length, embedding_dim))

    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)

    pos_enc = pos_enc[np.newaxis, ...]  # Add batch dimension

    return np.squeeze(pos_enc)

def padding_sequence_new(seq, window_size = 101):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size -seq_len
        new_seq = seq + [[0.25]*16] * gap_len
    return new_seq

def add_boundaries(seg,motif_size=4, num_features=16):
    bag_x=[]
    for i in range(motif_size-1):
        bag_x.append([0.25]*num_features)
    
    bag_x=bag_x+seg

    for i in range(motif_size-1):
        bag_x.append([0.25]*num_features) 
    
    return bag_x

def split_overlap_seq(seq, overlap_size = 50, num_feat=16, window_size=301, channel=2,motif_size=4):

    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        subseq=add_boundaries(subseq,motif_size, num_feat)
        bag_seqs.append(subseq)
        
    if num_ins == 0:
        seq1 = seq        
        pad_seq = padding_sequence_new(seq1, window_size)
        pad_seq=add_boundaries(pad_seq,motif_size, num_feat)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]            
            pad_seq = padding_sequence_new(seq1, window_size)
            pad_seq=add_boundaries(pad_seq,motif_size, num_feat)            
            bag_seqs.append(pad_seq)
    
    while len(bag_seqs)<channel:
        bag_seqs.append([[0.25]*num_feat]*(window_size+6))

    return bag_seqs

def generate_data(sequences,labels):
    X, Y=[],[]
    # print("Positive sequences..")
    for k, seq in enumerate(sequences):
        seq=seq.replace("T","U")
        seq_len=len(seq)
        x=np.zeros((seq_len,4))
        
        for i, b in enumerate(seq):
            x[i, BASE.index(b)]=1

        posi_encode = positional_encoding(seq_len, 12)
        x=np.concatenate([x,posi_encode], axis=-1)
        x=split_overlap_seq(seq=x.tolist(),
                            overlap_size = OVERLAP_SIZE, 
                            num_feat=IN_CHANNELS, 
                            window_size=WINDOW_SIZE, 
                            channel=MAX_SLICES,
                            motif_size=MOTIF_SIZE)
        X.append(x)
        Y.append(labels[k])

    return X, Y

def read_seq_graphprot(seq_file, label = 1):
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

base_path=os.path.join(os.getcwd(),'saved_models')
model_path=os.path.join(base_path,'PIONet')
if not os.path.isdir(model_path):
    os.makedirs(model_path)

def main(args):
    experiment=args.DATASET_NAME
    epochs=args.N_EPOCHS

    with open(os.path.join(os.getcwd(), 'pionet_output.txt'), 'a') as f:
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Starting Time:", formatted_date_time)
        f.write(f"\nStarting Time: {formatted_date_time}\n")

        data_path="data/RBP_24/GraphProt_CLIP_sequences"

        posi_train=os.path.join(data_path,f'{experiment}.train.positives.fa')
        nega_train=os.path.join(data_path,f'{experiment}.train.negatives.fa')    
        pos_seq_train, pos_label_train=read_seq_graphprot(posi_train, label = 1)
        neg_seq_train, neg_label_train=read_seq_graphprot(nega_train, label = 0)
        
        seq_train=pos_seq_train+neg_seq_train
        lab_train=pos_label_train+neg_label_train        
        seq_train, lab_train = shuffle(seq_train, lab_train)
        
        NUM_SEQ_EXP=len(seq_train)
        BATCH_SIZE=32
        NUM_BATCH=(NUM_SEQ_EXP//BATCH_SIZE)+1

        print(f'\nSize dataset, {experiment}: {NUM_SEQ_EXP}')
        f.write(f'Size of dataset {experiment}: {NUM_SEQ_EXP}\n\n')
        
        posi_test=os.path.join(data_path,f'{experiment}.ls.positives.fa')
        nega_test=os.path.join(data_path,f'{experiment}.ls.negatives.fa')
        pos_seq_test, pos_label_test=read_seq_graphprot(posi_test, label = 1)
        neg_seq_test, neg_label_test=read_seq_graphprot(nega_test, label = 0)
        
        seq_test=pos_seq_test+neg_seq_test
        lab_test=pos_label_test+neg_label_test  
        X_test, Y_test=generate_data(seq_test,lab_test) 
        
        batch=1
        start=(batch-1)*BATCH_SIZE
        end=start+BATCH_SIZE
                    
        print("Training in progress ..")    
        LR=0.001
        DROPOUT=0.25
        input_shape = (MAX_SLICES,WINDOW_SIZE+6,IN_CHANNELS)

        model = create_resnet_2D(input_shape)
        model.summary()
        
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        optimiser = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=optimiser, loss=loss_fn, metrics=['accuracy']) 

        
        # Define a metric to track accuracy
        train_acc_metric = tf.keras.metrics.BinaryAccuracy()

        best_auc_roc=0
        best_test_result=''
        best_epoch=0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # print("Training:")
            # Iterate over batches of the dataset
            for batch in tqdm.tqdm(range(1,NUM_BATCH+1)):
                if batch<NUM_BATCH:
                    start=(batch-1)*BATCH_SIZE
                    end=start+BATCH_SIZE         
                    X_train, Y_train=generate_data(seq_train[start:end],lab_train[start:end])         
                else:
                    start=(batch-1)*BATCH_SIZE
                    X_train, Y_train=generate_data(seq_train[start:], lab_train[start:])            
        
                   
                x_batch_train, y_batch_train = np.array(X_train), np.array(Y_train)

                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                
                # Compute gradients
                grads = tape.gradient(loss_value, model.trainable_weights)
                
                # Apply gradients to update weights
                optimiser.apply_gradients(zip(grads, model.trainable_weights))
                
                # Update accuracy metric
                train_acc_metric.update_state(y_batch_train, np.squeeze(logits))

            train_acc = train_acc_metric.result()

            # Reset metrics at the end of each epoch
            train_acc_metric.reset_state()
        
            
            #===================================Test++++++++++++++++++++++++++++++++

            # print("\nTesting:")      
            x_test= np.array(X_test)

            predictions = np.squeeze(model(x_test, training=False))
            
            # Convert the predictions to binary output (0 or 1)
            y_pred = (predictions > 0.5).astype(int)
            y_true= Y_test
            precision = metrics.precision_score(y_true, y_pred) #, zero_division=0)
            recall = metrics.recall_score(y_true, y_pred)
            accuracy = metrics.accuracy_score(y_true, y_pred)
            f1 = metrics.f1_score(y_true, y_pred)
            auc_roc = metrics.roc_auc_score(y_true, predictions)
            
            test_result=f'Epoch: {epoch+1}, Prec:{precision:.3f}, Recall: {recall:.3f}, Acc: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc_roc:.3f}'
            print (test_result)            
            f.write(f'{test_result}\n')

            
            if auc_roc> best_auc_roc:
                best_auc_roc=auc_roc
                best_test_result=f'Epoch: {epoch+1}, Prec:{precision:.3f}, Recall: {recall:.3f}, Acc: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc_roc:.3f}'
                model.save(f'{model_path}/pionet_{experiment}_{WINDOW_SIZE}.keras')

        print(f'\nMetrics:{experiment}')
        print("Best:",best_test_result)
        f.write(f'Best: {best_test_result}\n')
        
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Completion Time:", formatted_date_time)
        f.write(f"\nCompletion Time: {formatted_date_time}\n")
        
        f.close()

def parse_arguments(parser):
    parser.add_argument('--DATASET_NAME', type=str, default='ALKBH5_Baltz2012', help='Name of the fasta file')
    parser.add_argument('--N_EPOCHS', type=int, default=20)
    
    args = parser.parse_args()
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print ("Experiment:",args.DATASET_NAME,"| Epochs:",args.N_EPOCHS)
if __name__ == "__main__":
    main(args)

# Example to run in commandline
# python train_pionet.py --DATASET_NAME="ALKBH5_Baltz2012" --N_EPOCHS=10
import numpy as np
import torch

def inference(args, model, device, test_loader):
    model.eval()
    p_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()
            output = model(x)
            prob = torch.sigmoid(output)

            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

    p_all = np.concatenate(p_all)
    return p_all

def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm

def get_nt_height(pwm, height, norm):

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)

def sliding_window(sequence, score, k, step=1):
    # Slide a window of length k along the sequence with the given step size
    # For each window, extract the k-mer substring from the sequence
    # Compute the sum of scores corresponding to the positions of this k-mer
    # Store each k-mer and its total score into separate lists
    # Continue until the window reaches the end of the sequence
    # Return all extracted k-mers and their associated scores
    kmers = []
    kmer_scores = []
    for i in range(0, len(sequence) - k + 1, step):
        kmer = ''.join(sequence[i:i + k])
        kmers.append(kmer)
        kmer_score = sum(score[i + j] for j in range(k))
        kmer_scores.append(kmer_score)

    return kmers, kmer_scores

def sliding_window_ten(sequence, score, k, step=1, threshold=0.5):
    """
    Slides a window of length k across the sequence to extract k-mers.
    Calculates the cumulative score of each k-mer from the given score list.
    Finds the maximum score and sets a threshold based on it.
    Filters and returns only the k-mers whose scores are above the threshold.
    Returns None if no k-mer meets the threshold condition.
    """

    kmers = []
    kmer_scores = []
    for i in range(0, len(sequence) - k + 1, step):
        kmer = ''.join(sequence[i:i + k])
        kmers.append(kmer)
        kmer_score = sum(score[i + j] for j in range(k))
        kmer_scores.append(kmer_score)

    max_score = max(kmer_scores)
    score_threshold = max_score * threshold
    filtered_kmers = [kmers[i] for i in range(len(kmer_scores)) if kmer_scores[i] >= score_threshold]
    filtered_scores = [score for score in kmer_scores if score >= score_threshold]
    if not filtered_kmers:
        return None

    return filtered_kmers, filtered_scores

def sliding_window_six(sequence, score, k, step=1, threshold=0.5):
    """
    Slides a window of length k across the sequence to generate k-mers.
    Computes the cumulative score for each k-mer from the score list.
    Determines the maximum k-mer score and calculates a threshold value.
    Filters out k-mers whose scores are below the threshold.
    Returns the filtered k-mers and their scores, or None if none pass.
    """

    kmers = []
    kmer_scores = []
    for i in range(0, len(sequence) - k + 1, step):
        kmer = ''.join(sequence[i:i + k])
        kmers.append(kmer)
        kmer_score = sum(score[i + j] for j in range(k))
        kmer_scores.append(kmer_score)

    max_score = max(kmer_scores)
    score_threshold = max_score * threshold
    filtered_kmers = [kmers[i] for i in range(len(kmer_scores)) if kmer_scores[i] >= score_threshold]
    filtered_scores = [score for score in kmer_scores if score >= score_threshold]
    if not filtered_kmers:
        return None

    return filtered_kmers, filtered_scores

def normalize_array(arr):
    """
    Normalizes a 2D array by row.
    Computes the sum of absolute values in each row.
    Divides each element in a row by its row sum.
    Ensures each row sums to 1 in terms of absolute values.
    Returns the row-normalized array.
    """

    row_sums = np.sum(np.abs(arr), axis=1, keepdims=True)
    return arr / row_sums

def get_region(X, W):
    """
    Extracts high-saliency k-mer regions from input data.
    Normalizes the feature importance weights for sequence, one-hot, and structure features.
    Combines sequence and one-hot saliency scores to get a final importance score.
    Converts DNA 'T' bases to RNA 'U' in the sequence.
    Uses a sliding window of size 10 to compute k-mers and their cumulative scores.
    Returns the list of k-mers and their corresponding saliency scores.
    """

    seq = X[:1,:]
    seq_sal = normalize_array(W[:1, :])
    hot_sal = normalize_array(W[1:2, :])
    str_sal = normalize_array(W[2:, :])
    combined_sal = seq_sal+hot_sal
    sequence = seq[0]
    sequence = np.char.replace(sequence, 'T', 'U')
    score = combined_sal[0]
    mers,scores = sliding_window(sequence,score,10)

    return mers,scores

def get_region_no_nor(X, W, weight_fm, weight_hot):
    """
    Extracts high-saliency k-mer regions without normalizing all features.
    Normalizes sequence and one-hot saliency separately, then applies given weights.
    Converts DNA 'T' bases to RNA 'U' in the sequence.
    Computes k-mers and their scores using a sliding window of size 10.
    Filters k-mers based on a threshold and returns them along with their scores.
    Returns None if no k-mers pass the threshold.
    """

    seq = X[:1,:]
    seq_sal = normalize_array(W[:1, :])
    hot_sal = normalize_array(W[1:2, :])
    combined_sal = seq_sal*weight_fm+hot_sal*weight_hot
    sequence = seq[0]
    sequence = np.char.replace(sequence, 'T', 'U')
    score = combined_sal[0]
    result = sliding_window_ten(sequence,score,10)
    if result is None:
        return None
    else:
        mers,scores = result

    return mers,scores


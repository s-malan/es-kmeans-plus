"""
Extract downsampled acoustic word embeddings for a segment from the encoded audio.

Author: Herman Kamper, Simon Malan
Contact: kamperh@gmail.com, 24227013@sun.ac.za
Date: 2021, April 2024
"""

import numpy as np
import scipy.signal as signal

# here we downsample the MFCC or HuBERT embeddings to have a constant size for each segment
def downsample_utterance(features, segment, n):
    """
    Return the downsampled matrix with each row an embedding for a segment in
    the seglist.

    Parameters
    ----------
    features : tensor
        The encoded utterance to downsample
    segment : list (tuples)
        The segments (begin, end) to downsample
    n : int
        The number of samples
        default: 10

    Return
    ------
    embeddings : list (tensor)s
        The downsampled embeddings of the encoded utterance
    """

    embeddings = []
    for i, j in segment:
        y = features[i:j+1, :].T
        if j > features.shape[0]: # if the segment is longer than the utterance, stop at last frame
            if i < features.shape[0]:
                y = features[i:, :].T
            elif i == features.shape[0]:
                y = features[i-1:, :].T
        elif i == j: # if the segment is empty, add one frame
            y = features[i-1:j+1, :].T
        else:
            y = features[i:j+1, :].T
        
        y_new = signal.resample(y, n, axis=1).flatten("C")
        embeddings.append(y_new)
    return np.asarray(embeddings)

def downsample(utterances, segments, n=10):
    """
    Downsample each utterance.

    Parameters
    ----------
    utterances : list (tensor)
        The encoded utterances to downsample
    segments : list of lists (of tuples)
        The segments (begin, end) to downsample for each utterance
    n : int
        The number of samples
        default: 10

    Return
    ------
    downsampled_utterances : list (tensor)
        The downsampled embeddings of the encoded utterances
    """

    downsampled_utterances = []
    for utterance, segment in zip(utterances, segments):
        downsampled_utterances.append(downsample_utterance(utterance, segment, n))
    
    if len (downsampled_utterances) == 1:
        return downsampled_utterances[0]
    return downsampled_utterances
"""
Funtions used to evaluate word segmentation algorithm.

Author: Herman Kamper, Simon Malan
Contact: kamperh@gmail.com, 24227013@sun.ac.za
Date: March 2024
"""

import numpy as np

def eval_segmentation(seg, ref, strict=True, tolerance=1, continuous=False, num_seg=None, num_ref=None, num_hit=None):
    """
    Calculate number of hits of the segmentation boundaries with the ground truth boundaries.

    Parameters
    ----------
    seg : list of list of int
        The segmentation hypothesis word boundary frames for all utterances in the sample.
    ref : list of list of int
        The ground truth reference word boundary frames for all utterances in the sample.
    tolerance : int
        The number of offset frames that a segmentation hypothesis boundary can have with regards to a reference boundary and still be regarded as correct.
        default: 1 (10ms or 20ms for MFCCs with a frame shift of 10ms or for speech models)
    continuous : bool
        If True, return the number of segments, references, and hits instead of the evaluation metrics. This is to continue the evaluation over multiple samples.
        default: False
    num_seg : int
        The current number of segmentation boundaries.
        default: None
    num_ref : int
        The current number of reference boundaries.
        default: None
    num_hit : int
        The current number of hits.
        default: None

    Return
    ------
    output : (float, float, float)
        precision, recall, F-score.
    
    or if continuous:
        output : (int, int, int)
            num_seg, num_ref, num_hit
    """

    if not continuous:
        num_seg = 0 #Nf
        num_ref = 0 #Nref
        num_hit = 0 #Nhit
    
    assert len(seg) == len(ref) # Check if the number of utterances in the hypothesis and reference are the same
    for i_utterance in range(len(seg)): # for each utterance
        prediction = seg[i_utterance]
        ground_truth = ref[i_utterance]

        if len(prediction) > 0 and abs(prediction[-1] - ground_truth[-1]) <= tolerance: # if the last boundary is within the tolerance, delete it since it would have hit
            prediction = prediction[:-1]

        ground_truth = ground_truth[:-1] # Remove the last boundary of the reference if there is more than one boundary

        num_seg += len(prediction)
        num_ref += len(ground_truth)

        # count the number of hits
        for i_ref in ground_truth: # multiple ref can still hit on a single seg
            for i_seg in prediction:
                if abs(i_ref - i_seg) <= tolerance:
                    num_hit += 1
                    if strict: break # makes the evaluation strict, so that a reference boundary can only be hit once

    # Return the current counts
    return num_seg, num_ref, num_hit
    

def get_p_r_f1(num_seg, num_ref, num_hit):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.

    Parameters
    ----------
    num_seg : int
        The current number of segmentation boundaries.
        default: None
    num_ref : int
        The current number of reference boundaries.
        default: None
    num_hit : int
        The current number of hits.
        default: None

    Return
    ------
    output : (float, float, float)
        precision, recall, F-score.
    """

    # Calculate metrics, avoid division by zero:
    if num_seg == num_ref == 0:
        return 0, 0, -np.inf
    elif num_hit == 0:
        return 0, 0, 0
    
    if num_seg != 0:
        precision = float(num_hit/num_seg)
    else:
        precision = np.inf
    
    if num_ref != 0:
        recall = float(num_hit/num_ref)
    else:
        recall = np.inf
    
    if precision + recall != 0:
        f1_score = 2*precision*recall/(precision+recall)
    else:
        f1_score = -np.inf
    
    return precision, recall, f1_score

def get_os(precision, recall):
    """
    Calculates the over-segmentation; how many fewer/more boundaries are proposed compared to the ground truth.

    Parameters
    ----------
    precision : float
        How often word segmentation correctly predicts a word boundary.
    recall : float
        How often word segmentation's prediction matches a ground truth word boundary.

    Return
    ------
    output : float
        over-segmentation
    """

    if precision == 0:
        return -np.inf
    else:
        return recall/precision - 1
    
def get_rvalue(precision, recall):
    """
    Calculates the R-value; indicates how close (distance metric) the word segmentation performance is to an ideal point of operation (100% HR with 0% OS).

    Parameters
    ----------
    precision : float
        How often word segmentation correctly predicts a word boundary.
    recall : float
        How often word segmentation's prediction matches a ground truth word boundary.

    Return
    ------
    output : float
        R-Value
    """

    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)

    rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
    return rvalue
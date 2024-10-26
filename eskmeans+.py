"""
Main script to segment audio using ES-KMeans+, and evaluate the resulting segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import numpy as np
import argparse
import random
from pathlib import Path
import os
from tqdm import tqdm
from utils import data_process
from wordseg import landmark_seg, segment_new
from sklearn.decomposition import PCA

def get_data(data, args, speaker, segmenter):
    """
    Sample data and get the landmarks and  for the sampled audio file.

    Parameters
    ----------
    data : Features
        The data object, containing the features.
    args : Namespace
        The arguments for the script.
    speaker : list (str)
        The speaker being processed.
        default: [None] (speaker-independet)
    segmenter : ESKmeans
        The ESKmeans object, containing all information of the selected features.

    Return
    ------
    samples : list (str)
        The list of the sampled file paths to the features of the utterances.
    wavs : list (str)
        The list of the sampled file paths to the audio files of the utterances.
    landmarks : list (list (int))
        The landmarks (in frames) of the utterances.
    """

    samples, wavs = data.sample_features(speaker) # sample file paths from the features
       
    pca = None
    if args.model not in ["mfcc", "melspec"]:
        print('Fitting PCA')
        pca = PCA(n_components=250)
        pca.fit(np.concatenate(data.load_features(random.sample(samples, int(0.8*len(samples)))), axis=0))

    if len(samples) == 0:
        print('No utterances to segment, sampled only one utterance that only has one frame.')
        exit()
    
    # Get landmarks
    landmarks, segments, lengths = get_landmarks(data, args, wavs, segmenter.n_slices_max)

    durations = []
    vec_ids = []
    boundaries = []
    for landmark, segment, length in tqdm(zip(landmarks, segments, lengths), desc='Getting utterance data'):

        # Get durations and active segments
        duration = get_durations(landmark, segmenter.min_duration)
        vec_id = get_vec_ids(length, segmenter.n_slices_max)
        boundary = segmenter.get_boundaries(length)
        durations.append(duration)
        vec_ids.append(vec_id)
        boundaries.append(boundary)

    # use fixed K_max for zrc2017
    # K_max = 43000 # TODO 43000 for zrc english!!!
    # K_max = 29000 # TODO 29000 for zrc french!!!
    # K_max = 3000 # TODO 3000 for zrc mandarin!!!
    # K_max = 29000 # TODO 29000 for zrc german!!!
    # K_max = 3500 # TODO 3500 for zrc wolof!!!
    # K_max = 13967 # TODO 13967 for librispeech!!!
    K_max = args.k_max

    # load the utterance data into the ESKMeans segmenter
    segmenter.set_data(data, samples, landmarks, segments, lengths, durations, vec_ids, boundaries, K_max, pca)

    return samples, wavs, landmarks

def get_landmarks(data, args, wavs, n_slices_max):
    """
    Get the landmarks and their lengths for the utterances.

    Parameters
    ----------
    data : Features
        The data object, containing the features.
    args : Namespace
        The arguments for the script.
    wavs : list (str)
        The list of the sampled file paths to the audio files of the utterances.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be calculated.

    Return
    ------
    landmarks : list (list (int))
        The landmarks (in frames) of the utterances.   
    segments : list (list (tuple (int, int)))
        The segments (tuple of neighbouring landmarks) of the utterances.
    lengths : list (int)
        The number of landmarks per utterance.
    """

    landmarks = []
    segments = []
    lengths = []
    for wav in tqdm(wavs, desc="Getting landmarks"):

        # Get and load/save landmarks
        landmark_details = os.path.split(wav)
        landmark_dir = Path(args.load_landmarks / landmark_details[-1]).with_suffix(".list")
        with open(landmark_dir) as f:
            landmark = []
            for line in f:
                landmark.append(float(line.strip())) # loaded into frames
            landmark = data.get_frame_num(np.array(landmark)).astype(np.int32).tolist()
            if landmark[-1] == 0:
                landmark = [1] # fix rounding error (0.5 -> 0.0)
            landmarks.append(landmark)

        segments.append(landmark_seg.get_segments(landmarks[-1], max_span = n_slices_max))
        lengths.append(len(landmarks[-1]))
    
    return landmarks, segments, lengths

def get_vec_ids(n_slices, n_slices_max):
    """
    Get the vector ids for an utterance. The indices of the embeddings between different landmarks. 

    Parameters
    ----------
    n_slices : list (int)
        The number of landmarks per utterance.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be calculated.

    Return
    ------
    vec_ids : list (int)
        The vector ids for the utterance.
    """

    vec_ids = -1*np.ones(int((n_slices**2 + n_slices)/2), dtype=int)
    i_embed = 0
    for cur_start in range(n_slices):
        for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
            cur_end += 1
            t = cur_end
            i = t*(t - 1)/2
            vec_ids[int(i + cur_start)] = i_embed
            i_embed += 1

    return vec_ids

def get_durations(landmarks, min_duration):
    """
    Get the duration of each segment of an utterance in frames.

    Parameters
    ----------
    landmarks : list (list (int))
        The landmarks (in frames) of the utterances. 
    min_duration : int
        Minimum duration (in frames) of a segment.

    Return
    ------
    duration : list (int)
        The duration (in frames) of each of the segments. 
    """

    duration = []
    landmarks_ = [0] + landmarks
    N = len(landmarks_)
    duration = -1*np.ones(int(((N - 1)**2 + (N - 1))/2), dtype=int)
    j = 0
    for t in range(1, N):
        for i in range(t):
            if t - i > N - 1:
                j += 1
                continue
            duration[j] = landmarks_[t] - landmarks_[i]
            j += 1
    
    if not (min_duration == 0 or len(duration) == 1):
        cur_duration_vec = np.array(duration, dtype=np.float64)
        cur_duration_vec[cur_duration_vec < min_duration] = -1
        if np.all(cur_duration_vec == -1):
            cur_duration_vec[np.argmax(duration)] = np.max(duration)
        duration = cur_duration_vec

    return duration.astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES-KMeans+ wordseg.")
    parser.add_argument(
        "model",
        help="available models",
        default="mfcc",
    )
    parser.add_argument(
        "layer", # -1 for no layer
        type=int,
    )
    parser.add_argument(
        "wav_dir",
        metavar="wav-dir",
        help="path to the audio waveform directory.",
        type=Path,
    )
    parser.add_argument(
        "feature_dir",
        metavar="feature-dir",
        help="path to the speech feature directory.",
        type=Path,
    )
    parser.add_argument(
        "load_landmarks",
        help="root landmark directory to load landmarks.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "save_segments",
        help="root directory to save word boundaries.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "k_max",
        help="maximum number of clusters.",
        type=int,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio waveform files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--sample_size",
        metavar="sample-size",
        help="number of features to sample (-1 to sample all available data).",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--speaker",
        help="Speaker list if speaker dependent sampling must be done.",
        default=None,
        type=Path,
    )
    args = parser.parse_args()

    if args.model in ["mfcc", "melspec"]:
        mfcc = True
        frame_len = 10
    else:
        mfcc = False
        frame_len = 20

    # ~~~~~~~~~~ Sample a audio files and features ~~~~~~~~~~
    data = data_process.Features(wav_dir=args.wav_dir, root_dir=args.feature_dir, model_name=args.model, layer=args.layer, 
                                 extension=args.extension, num_files=args.sample_size, frames_per_ms=frame_len)

    if args.speaker is not None:
        speakerlist = data.get_speakers(args.speaker)
    else:
        speakerlist = [None]

    random.seed(42)
    np.random.seed(42)
    n_slices_max = 4
    if mfcc:
        min_duration = 5
    else:
        min_duration = 3

    for speaker in tqdm(speakerlist, desc="Speaker"):
        
        # ~~~~~~~~~~ Sample data and get the landmarks for the sampled audio file ~~~~~~~~~~
        segmenter = segment_new.ESKmeans( # setup base ESKmeans object
            p_boundary_init=0.5, n_slices_max=n_slices_max, min_duration=min_duration)
        
        samples, wavs, landmarks = get_data(data, args, speaker, segmenter)

        # ~~~~~~~~~~ ES-KMeans+ segmentation ~~~~~~~~~~
        _, classes = segmenter.segment(n_iterations=5)

        # ~~~~~~~~~~~~~~~~~~~ Save utterance segments and assignments ~~~~~~~~~~~~~~~~~~~~
        seg_list = []
        for i in tqdm(range(segmenter.D), desc='Getting boundary frames and classes'): # for each utterance
            segmentation_frames = landmarks[i]*segmenter.boundaries[i]
            segmentation_frames = [x for x in segmentation_frames if x != 0]
            seg_list.append(segmentation_frames)
            
            if len(classes[i]) == 1:
                class_i = classes[i]
            else:
                class_i = [x for x in classes[i] if x != -1]
            save_dir = (args.save_segments / os.path.split(wavs[i])[-1]).with_suffix(".list")
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(save_dir, "w") as f:
                for t, c in zip(segmentation_frames, class_i):
                    f.write(f"{data.get_sample_second(t)} {int(c)}\n")
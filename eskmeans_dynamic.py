"""
Main script to segment audio using ES-KMeans, and evaluate the resulting segmentation.

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
from wordseg import landmark_seg, downsample, evaluate, segment_dynamic
from sklearn.decomposition import PCA

# sys.path.append(str(Path("..")/".."/"src"/"eskmeans"/"utils")) use something like this to import the data_process scripts

def get_data(data, args, speaker, segmenter, proportion):

    samples, wavs = data.sample_embeddings(speaker) # sample file paths from the feature embeddings
       
    pca = None
    if args.model not in ["mfcc", "melspec"]:
        print('Fitting PCA')
        pca = PCA(n_components=15)
        pca.fit(np.concatenate(data.load_embeddings(samples), axis=0)) # load all embeddings and fit PCA, then use this pca to to later transform the embeddings

    if len(samples) == 0:
        print('No embeddings to segment, sampled a file with only one frame.')
        exit()
    
    # Get landmarks
    landmarks, segments, lengths, n_landmarks = get_landmarks(data, args, wavs, segmenter.n_slices_max)

    durations = []
    vec_ids = []
    boundaries = []
    for landmark, length in tqdm(zip(landmarks, lengths), desc='Getting utterance data'): # for each utterance get the initial data (landmarks, segments, lengths)

        # Get durations and active segments
        duration = get_durations(landmark, segmenter.min_duration)
        vec_id = get_vec_ids(length, segmenter.n_slices_max)
        boundary = segmenter.get_boundaries(length)
        durations.append(duration)
        vec_ids.append(vec_id)
        boundaries.append(boundary)

    K_max = int(np.floor(proportion * n_landmarks))

    # use fixed K_max for zrc2017
    # K_max = 43000 # TODO 43000 for zrc!!!

    # load the utterance data into the ESKMeans segmenter
    segmenter.set_data(data, samples, landmarks, segments, lengths, durations, vec_ids, boundaries, K_max, pca)

    return samples, wavs, landmarks

def get_landmarks(data, args, wavs, n_slices_max):
    landmarks = []
    segments = []
    lengths = []
    n_landmarks = 0
    for wav in tqdm(wavs, desc="Getting landmarks"): # for each utterance
        
        # Get and load/save landmarks
        landmark_details = os.path.split(wav)

        if args.load_landmarks is not None:
            landmark_dir = Path(args.load_landmarks / landmark_details[-1]).with_suffix(".list")
            with open(landmark_dir) as f:
                landmark = []
                for line in f:
                    landmark.append(float(line.strip())) # loaded into frames
                landmarks.append(data.get_frame_num(np.array(landmark)).astype(np.int32).tolist())
        else:
            landmark_root_dir = os.path.split(args.embeddings_dir)
            landmark_root_dir = os.path.join(os.path.commonpath([args.embeddings_dir, wav]), 'segments', 'sylseg', os.path.split(landmark_root_dir[0])[-1], landmark_root_dir[-1])
            if args.layer != -1:
                landmark_dir = Path(os.path.join(landmark_root_dir, args.model, str(args.layer), landmark_details[-1])).with_suffix(".list")
            else:
                landmark_dir = Path(os.path.join(landmark_root_dir, args.model, landmark_details[-1])).with_suffix(".list")

            if os.path.isfile(landmark_dir):
                with open(landmark_dir) as f: # if the landmarks are already saved to a file
                    landmark = []
                    for line in f:
                        landmark.append(data.get_frame_num(line.strip())) # load into frames
                    landmarks.append(landmark)
            else:
                landmarks.append(data.get_frame_num(landmark_seg.get_boundaries(wav, fs=16000)).astype(np.int32).tolist()[1:]) # get the boundaries in frames
                landmark_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(landmark_dir, "w") as f: # save the landmarks to a file
                    for l in landmarks[-1]:
                        f.write(f"{data.get_sample_second(l)}\n") # save in seconds

        segments.append(landmark_seg.get_segments(landmarks[-1], max_span = n_slices_max))
        lengths.append(len(landmarks[-1]))
        n_landmarks += lengths[-1]
    
    return landmarks, segments, lengths, n_landmarks

def get_vec_ids(n_slices, n_slices_max):
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

def get_durations(landmarks, min_duration): # duration of each segment IN FRAMES
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
        # cur_duration_vec[cur_duration_vec < min_duration] = -np.nan
        cur_duration_vec[cur_duration_vec < min_duration] = -1
        # if np.all(np.isnan(cur_duration_vec)):
        if np.all(cur_duration_vec == -1):
            cur_duration_vec[np.argmax(duration)] = np.max(duration)
        duration = cur_duration_vec

    return duration.astype(int)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES-KMeans wordseg.")
    parser.add_argument(
        "model",
        help="available models (MFCCs)",
        choices=["mfcc", "hubert_shall", "hubert_fs", "w2v2_hf"],
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
        "embeddings_dir",
        metavar="embeddings-dir",
        help="path to the embeddings directory.",
        type=Path,
    )
    parser.add_argument(
        "alignments_dir",
        metavar="alignments-dir",
        help="path to the alignments directory.",
        type=Path,
    )
    parser.add_argument(
        "sample_size",
        metavar="sample-size",
        help="number of embeddings to sample (-1 to sample all available data).",
        type=int,
    )
    parser.add_argument(
        "--wav_format",
        help="extension of the audio waveform files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--align_format",
        help="extension of the alignment files (defaults to .TextGrid).",
        default=".TextGrid",
        type=str,
    )
    parser.add_argument(
        "--load_landmarks",
        help="root landmark directory to load landmarks.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--save_segments",
        help="root directory to save word boundaries.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--speaker",
        help="Speaker list if speaker dependent sampling must be done.",
        default=None,
        type=Path,
    )
    parser.add_argument( # optional argument to make the evaluation strict
        '--strict',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.model in ["mfcc", "melspec"]:
        mfcc = True
        frame_len = 10
    else:
        mfcc = False
        frame_len = 20

    # ~~~~~~~~~~ Sample a audio file and its alignments ~~~~~~~~~~
    data = data_process.Features(wav_dir=args.wav_dir, root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, wav_format=args.wav_format, alignment_format=args.align_format, num_files=args.sample_size, frames_per_ms=frame_len)
    # python3 eskmeans_dynamic.py mfcc -1 /media/hdd/data/buckeye_segments/test /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/buckeye_segments/buckeye_test_speakers.list --strict
    # python3 eskmeans_dynamic.py mfcc -1 /media/hdd/data/zrc/zrc2017_train_segments/english /media/hdd/embeddings/zrc/zrc2017_train_segments/english /media/hdd/data/zrc_alignments/zrc2017_train_alignments/english -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/zrc/zrc2017-train-dataset/index.json --strict
    # python3 eskmeans_dynamic.py hubert_shall 10 /media/hdd/data/librispeech /media/hdd/embeddings/librispeech /media/hdd/data/librispeech_alignments -1  --load_landmarks=/media/hdd/segments/tti_wordseg/librispeech/dev_clean/hubert_shall/10 --strict
    # --load_landmarks=/media/hdd/segments/sylseg/buckeye/test/mfcc OR /media/hdd/segments/sylseg/zrc2017_train_segments/english/mfcc
    # --load_landmarks=/media/hdd/segments/tti_wordseg/buckeye/test/hubert_shall/10 OR /media/hdd/segments/tti_wordseg/zrc2017_train_segments/english/hubert_shall/10
    # --save_segments=/media/hdd/segments/eskmeans/sylseg/buckeye/test OR /media/hdd/segments/eskmeans/sylseg/zrc2017_train_segments/english
    # --save_segments=/media/hdd/segments/eskmeans/tti/buckeye/test OR /media/hdd/segments/eskmeans/tti/zrc2017_train_segments/english

    if args.speaker is not None:
        speakerlist = data.get_speakers(args.speaker)
    else:
        speakerlist = [None]

    num_hit = 0
    num_ref = 0
    num_seg = 0
    for speaker in tqdm(speakerlist, desc="Speaker"):
        if mfcc:
            min_duration = 12 # TODO 5 for zrc!!! # 12 frames is 120ms minimum duration
        else:
            min_duration = 6 # TODO 3 for zrc!!! # 6 frames is 120ms minimum duration
        
        # ~~~~~~~~~~ Sample data and get the landmarks for the sampled audio file ~~~~~~~~~~
        print('1. Sampling speaker data:', speaker)
        random.seed(42)
        np.random.seed(42)
        n_slices_max = 6 # TODO 4 for zrc!!! # max number of landmarks (syllables) a segment can span, n_landmarks_max in notebook
        proportion = 0.2

        segmenter = segment_dynamic.ESKmeans_Dynamic( # setup base ESKmeans object
            p_boundary_init=1.0, n_slices_max=n_slices_max, min_duration=min_duration
            )
        # get data for all utterances without saving the embeddings
        samples, wavs, landmarks = get_data(data, args, speaker, segmenter, proportion)

        # ~~~~~~~~~~ ES-KMeans segmentation ~~~~~~~~~~
        print('2. ES-KMeans')
        _, classes = segmenter.segment(n_iterations=10) # TODO 5 for zrc!!!

        seg_list = []
        for i in tqdm(range(segmenter.D), desc='Getting boundary frames and classes'): # for each utterance
            segmentation_frames = landmarks[i]*segmenter.boundaries[i]
            segmentation_frames = [x for x in segmentation_frames if x != 0]
            seg_list.append(segmentation_frames)
            
            if args.save_segments is not None: # save the segments
                if len(classes[i]) < 2: # if there is only one class
                    class_i = classes[i]
                else:
                    class_i = [x for x in classes[i] if x != -1] # get classes that are not -1
                save_dir = (args.save_segments / os.path.split(wavs[i])[-1]).with_suffix(".list")
                save_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(save_dir, "w") as f:
                    for t, c in zip(segmentation_frames, class_i):
                        f.write(f"{data.get_sample_second(t)} {c}\n") # save in seconds
        
        # ~~~~~~~~~~ Evaluation ~~~~~~~~~~
        if mfcc:
            tol = 2 # 20ms
        else:
            tol = 1 # 20ms
        
        # Get alignements
        # get the paths to the alignments corresponding to the sampled embeddings
        data.set_alignments(files=data.get_alignment_paths(files=samples)) # set the text, start and end attributes of the alignment files
        alignment_end_frames = [alignment.end for alignment in data.alignment_data[:]]
        num_seg, num_ref, num_hit = evaluate.eval_segmentation(seg_list, alignment_end_frames, strict=True, tolerance=tol, continuous=True, num_seg=num_seg, num_ref=num_ref, num_hit=num_hit)

    precision, recall, f1_score = evaluate.get_p_r_f1(num_seg, num_ref, num_hit)
    r_value = evaluate.get_rvalue(precision, recall)

    print(f"Precision: {precision:.6f}, Recall: {recall:.6f}, F1-score: {f1_score:.6f}, R-value: {r_value:.6f}")
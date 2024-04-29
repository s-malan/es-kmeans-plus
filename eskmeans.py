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
from wordseg import landmark_seg, downsample, segment, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES-KMeans wordseg.")
    parser.add_argument(
        "model",
        help="available models (MFCCs)",
        choices=["mfcc"],
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
        "--speaker",
        help="True if speaker dependent sampling must be done.",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument( # optional argument to make the evaluation strict
        '--strict',
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

# ~~~~~~~~~~ Sample a audio file and its alignments ~~~~~~~~~~
data = data_process.Features(wav_dir=args.wav_dir, root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, wav_format=args.wav_format, alignment_format=args.align_format, num_files=args.sample_size, frames_per_ms=10)
# python3 eskmeans.py mfcc -1 /media/hdd/data/buckeye_segments/dev /media/hdd/embeddings/buckeye/dev /media/hdd/data/buckeye_alignments/dev -1 --wav_format=.wav --align_format=.txt --speaker --strict

if args.speaker:
    speakerlist = data.get_speakers()
else:
    speakerlist = [None]

num_hit = 0
num_ref = 0
num_seg = 0
for speaker in tqdm(speakerlist, desc="Speaker"): # TODO put each step in a function
    sample, wavs = data.sample_embeddings(speaker) # sample from the feature embeddings
    embeddings = data.load_embeddings(sample) # load the sampled embeddings
    # norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

    index_del = []
    for i, embedding in enumerate(embeddings): # delete embeddings with only one frame
        if embedding.shape[0] == 1:
            index_del.append(i)
    
    for i in sorted(index_del, reverse=True):
        del sample[i]
        del embeddings[i]
        # del norm_embeddings[i]

    if len(sample) == 0:
        print('No embeddings to segment, sampled a file with only one frame.')
        exit()

    print('1. Sampling speaker:', speaker)
    # print(embeddings[0].shape)
    # print(norm_embeddings[0].shape)

    # Get alignements
    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files
    # print(data.alignment_data[0].text, data.alignment_data[0].end)

# ~~~~~~~~~~ Get the landmarks for the sampled audio file ~~~~~~~~~~

    print('2. Landmarks') # TODO maybe save and then load again
    n_slices_max = 6 # max number of landmarks (syllables) a segment can span, n_landmarks_max in notebook
    landmarks = []
    segments = []
    lengths = []
    n_landmarks = 0
    for wav in tqdm(wavs): # for each utterance
        
        # Get and load/save landmarks TODO add after segments (eskmeans)
        landmark_root_dir = os.path.split(args.embeddings_dir)
        landmark_details = os.path.split(wav)
        landmark_root_dir = os.path.join(os.path.commonpath([args.embeddings_dir, wav]), 'segments', 'eskmeans', os.path.split(landmark_root_dir[0])[-1], landmark_root_dir[-1])
        if args.layer != -1:
            landmark_dir = os.path.join(landmark_root_dir, args.model, str(args.layer), landmark_details[-1]).with_suffix(".list")
        else:
            landmark_dir = Path(os.path.join(landmark_root_dir ,args.model, landmark_details[-1])).with_suffix(".list")

        if os.path.isfile(landmark_dir):
            with open(landmark_dir) as f: # if the landmarks are already saved to a file
                landmark = []
                for line in f:
                    landmark.append(int(line.strip()))
                landmarks.append(landmark)
        else:
            landmarks.append(np.ceil(landmark_seg.get_boundaries(wav, fs=16000)*100).astype(np.int32).tolist()[1:]) # get the boundaries
            landmark_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(landmark_dir, "w") as f: # save the landmarks to a file
                for l in landmarks[-1]:
                    f.write(f"{l}\n")

        segments.append(landmark_seg.get_segments(landmarks[-1], max_span = n_slices_max))
        lengths.append(len(landmarks[-1]))
        n_landmarks += lengths[-1]
    print('\n')

# ~~~~~~~~~~ Test downsampling each segment of each utterance ~~~~~~~~~~

    print('3. Downsampling ex. (do one-for-one later)')
    downsampled_utterances = downsample.downsample(embeddings, segments, n=10)
    # print(downsampled_utterances[-1].shape)

# ~~~~~~~~~~ ES-KMeans segmentation ~~~~~~~~~~

    print('4. ES-KMeans')
    print('Herman\'s code')

    vec_ids = []
    for i_utt in range(len(downsampled_utterances)):
        # Vector IDs: `vec_ids[i:i+ t]` contains the IDs of embedding[0:t] up to embedding[t - 1:t], with i = t(t - 1)/2
        n_slices = lengths[i_utt] # number of landmarks in the utterance
        vec_id = -1*np.ones(int((n_slices**2 + n_slices)/2), dtype=int)
        i_embed = 0
        for cur_start in range(n_slices):
            for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
                cur_end += 1
                t = cur_end
                i = t*(t - 1)/2
                vec_id[int(i + cur_start)] = i_embed
                i_embed += 1
        vec_ids.append(vec_id.tolist())
    
    # for i_utt in range(len(downsampled_utterances)):
    #     print(len(vec_ids[i_utt]), len(segments[i_utt]), lengths[i_utt])
    #     print((vec_ids[i_utt]), (segments[i_utt]), lengths[i_utt])

    durations = [] # !! duration of each segment IN FRAMES (10ms units)
    for i_utt in range(len(downsampled_utterances)):
        landmarks_ = [0] + landmarks[i_utt]
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
        durations.append(duration.tolist())
    
    random.seed(2)
    np.random.seed(2)

    # K_max = 5 # max number of clusters
    proportion = 0.21
    K_max = int(np.floor(proportion * n_landmarks)) #https://github.com/kamperh/bucktsong_eskmeans/blob/master/segmentation/ksegment.py#L111
    segmenter = segment.ESKmeans_Herman(
        K_max, downsampled_utterances, vec_ids, durations, landmarks, lengths,
        p_boundary_init=0.5, n_slices_max=n_slices_max, min_duration=20 # 20 frames is 200ms minimum duration
        )
    # TODO pre-train the kmeans model a bit https://github.com/kamperh/bucktsong_eskmeans/blob/master/segmentation/ksegment.py#L150
    segmenter.segment(n_iterations=10)

    seg_list = []
    for i in range(segmenter.D): # for each utterance
        # print(wavs[i], segmenter.get_unsup_transcript_i(i), segmenter.get_segmented_embeds_i(i), landmarks[i]*segmenter.boundaries[i][0:len(landmarks[i])], landmarks[i])
        segmentation_frames = landmarks[i]*segmenter.boundaries[i][0:len(landmarks[i])]
        segmentation_frames = [x for x in segmentation_frames if x != 0]
        seg_list.append(segmentation_frames)
    
    # Evaluate
    alignment_end_times = [alignment.end for alignment in data.alignment_data[:]]
    alignment_end_frames = []
    for alignment_times in alignment_end_times:
        alignment_frames = [data.get_frame_num(end_time) for end_time in alignment_times]
        alignment_end_frames.append(alignment_frames)
    num_seg, num_ref, num_hit = evaluate.eval_segmentation(seg_list, alignment_end_frames, strict=True, tolerance=2, continuous=True, num_seg=num_seg, num_ref=num_ref, num_hit=num_hit)

# ~~~~~~~~~~ Evaluate the segmentation ~~~~~~~~~~ TODO add NED and WER
precision, recall, f1_score = evaluate.get_p_r_f1(num_seg, num_ref, num_hit)
r_value = evaluate.get_rvalue(precision, recall)

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1_score:.3f}, R-value: {r_value:.3f}")
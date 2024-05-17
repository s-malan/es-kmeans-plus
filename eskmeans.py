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
from wordseg import segment_CSC, segment_FAISS

import time
# sys.path.append(str(Path("..")/".."/"src"/"eskmeans"/"utils")) use something like this to import the data_process scripts

def sample_utt(data, speaker):
    sample, wavs = data.sample_embeddings(speaker) # sample from the feature embeddings
    # sample = ['/media/hdd/embeddings/buckeye/test/mfcc/s01_01a_004731-005014.npy'] # TODO remove
    embeddings = data.load_embeddings(sample) # load the sampled embeddings

    index_del = []
    for i, embedding in enumerate(embeddings): # delete embeddings with only one frame
        if embedding.shape[0] == 1:
            index_del.append(i)
    
    for i in sorted(index_del, reverse=True):
        del sample[i]
        del embeddings[i]

    if len(sample) == 0:
        print('No embeddings to segment, sampled a file with only one frame.')
        exit()

    # Get alignements
    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files

    return sample, wavs, embeddings

def get_landmarks(data, args, wavs, n_slices_max):
    landmarks = []
    segments = []
    lengths = []
    n_landmarks = 0
    for wav in tqdm(wavs): # for each utterance
        
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
                landmark_dir = os.path.join(landmark_root_dir, args.model, str(args.layer), landmark_details[-1]).with_suffix(".list")
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

def get_vec_ids(downsampled_utterances, lengths, n_slices_max):
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

    return vec_ids

def get_durations(downsampled_utterances, landmarks):
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

    return durations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES-KMeans wordseg.")
    parser.add_argument(
        "model",
        help="available models (MFCCs)",
        choices=["mfcc", "hubert_shall"],
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
    # python3 eskmeans.py mfcc -1 /media/hdd/data/buckeye_segments/test /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/buckeye_segments/buckeye_test_speakers.list --strict
    # python3 eskmeans.py mfcc -1 /media/hdd/data/zrc/zrc2017_train_segments/english /media/hdd/embeddings/zrc/zrc2017_train_segments/english /media/hdd/data/zrc_alignments/zrc2017_train_alignments/english -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/zrc/zrc2017-train-dataset/index.json --strict
    # --load_landmarks=/media/hdd/segments/sylseg/buckeye/test/mfcc OR /media/hdd/segments/sylseg/zrc2017_train_segments/english/mfcc
    # --load_landmarks=/media/hdd/segments/tti_wordseg/buckeye/test/hubert_shall/10 OR /media/hdd/segments/tti_wordseg/zrc2017_train_segments/english/hubert_shall/10
    # --save_segments=/media/hdd/segments/eskmeans/sylseg/buckeye/test OR /media/hdd/segments/eskmeans/sylseg/zrc2017_train_segments/english
    # --save_segments=/media/hdd/segments/eskmeans/tti/buckeye/test OR /media/hdd/segments/eskmeans/tti/zrc2017_train_segments/english

    if args.speaker is not None:
        speakerlist = data.get_speakers(args.speaker)
    else:
        speakerlist = [None]
    
    # speakerlist = [None]

    num_hit = 0
    num_ref = 0
    num_seg = 0
    for speaker in tqdm(speakerlist, desc="Speaker"):

        print('1. Sampling speaker:', speaker)
        sample, wavs, embeddings = sample_utt(data, speaker)

        # # TODO remove below
        # seg_list = []
        # for wav in wavs:
        #     dir = Path(Path("/home/simon/Downloads/eskmeans/buckeye/test/intervals") / Path(wav.split('/')[-1].split('.')[0])).with_suffix(".txt")
        #     # dir = Path('/home/simon/Downloads/eskmeans/buckeye/test/intervals/s01_01a_004731-005014.txt')
        #     with open(dir) as f:
        #         segments = []
        #         for line in f:
        #             _, seg, _ = line.split()
        #             segments.append(int(seg))
        #     seg_list.append(segments)
        #     # break

        # ~~~~~~~~~~ Get the landmarks for the sampled audio file ~~~~~~~~~~
        print('2. Landmarks')
        n_slices_max = 6 # (6, 4/4*) # max number of landmarks (syllables) a segment can span, n_landmarks_max in notebook
        landmarks, segments, lengths, n_landmarks = get_landmarks(data, args, wavs, n_slices_max)

        
        # ~~~~~~~~~~ Test downsampling each segment of each utterance ~~~~~~~~~~
        print('3. Downsampling')
        if mfcc:
            min_duration = 0 # 20 frames is 200ms minimum duration # TODO change back to 20
            downsampled_utterances = downsample.downsample(embeddings, segments, n=10)
        else:
            min_duration = 10 # 10 frames is 200ms minimum duration
            downsampled_utterances = []
            for embedding_i, segment_i in zip(embeddings, segments): # for each utterance
                downsampled_utterances.append(np.stack([embedding_i[a:b, :].mean(0) for a, b in segment_i])) # 1268; ndarray(27, 768)

        # ~~~~~~~~~~ ES-KMeans segmentation ~~~~~~~~~~
        print('4. ES-KMeans')
        random.seed(42)
        np.random.seed(42)

        vec_ids = get_vec_ids(downsampled_utterances, lengths, n_slices_max)
        durations = get_durations(downsampled_utterances, landmarks)

        proportion = 0.75 # (0.2, 0.1/0.2*)
        K_max = int(np.floor(proportion * n_landmarks)) #https://github.com/kamperh/bucktsong_eskmeans/blob/master/segmentation/ksegment.py#L111

        # segmenter = segment.ESKmeans_Herman(
        #     K_max, downsampled_utterances, vec_ids, durations, landmarks, lengths,
        #     p_boundary_init=1.0, n_slices_max=n_slices_max, min_duration=min_duration # (1.0 -> 0.1, 0.5/0.0*)
        #     )
        
        # For Cluster -> Segment -> Cluster:
        # segmenter = segment_CSC.ESKmeans_CSC()
        #     K_max, downsampled_utterances, vec_ids, durations, landmarks, lengths,
        #     p_boundary_init=1.0, n_slices_max=n_slices_max, min_duration=min_duration
        #     )

        # For Cluster -> Segment -> Cluster using FAISS KMeans:
        segmenter = segment_FAISS.ESKmeans_FAISS(
            K_max, downsampled_utterances, vec_ids, durations, landmarks, lengths,
            p_boundary_init=1.0, n_slices_max=n_slices_max, min_duration=min_duration
            )
        
        start_time = time.time()
        segmenter.segment(n_iterations=10) # (10, 10/5*)
        print(f"Time taken: {time.time() - start_time}s")

        seg_list = []
        for i in range(segmenter.D): # for each utterance
            segmentation_frames = landmarks[i]*segmenter.boundaries[i][0:len(landmarks[i])]
            segmentation_frames = [x for x in segmentation_frames if x != 0]
            seg_list.append(segmentation_frames)
            # classes = segmenter.get_unsup_transcript_i(i) # TODO use FAISS to get the classes
            
            if args.save_segments is not None: # save the segments
                classes = segmenter.get_unsup_transcript_i(i)
                save_dir = (args.save_segments / os.path.split(wavs[i])[-1]).with_suffix(".list")
                save_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(save_dir, "w") as f:
                    for t, c in zip(segmentation_frames, classes):
                        f.write(f"{data.get_sample_second(t)} {c}\n") # save in seconds
        
        # ~~~~~~~~~~ Evaluation ~~~~~~~~~~
        if mfcc:
            tol = 2 # 20ms
        else:
            tol = 1 # 20ms
        alignment_end_frames = [alignment.end for alignment in data.alignment_data[:]]
        num_seg, num_ref, num_hit = evaluate.eval_segmentation(seg_list, alignment_end_frames, strict=True, tolerance=tol, continuous=True, num_seg=num_seg, num_ref=num_ref, num_hit=num_hit)

    precision, recall, f1_score = evaluate.get_p_r_f1(num_seg, num_ref, num_hit)
    r_value = evaluate.get_rvalue(precision, recall)

    print(f"Precision: {precision:.6f}, Recall: {recall:.6f}, F1-score: {f1_score:.6f}, R-value: {r_value:.6f}")
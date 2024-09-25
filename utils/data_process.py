"""
Sample audio features and normalize them.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import numpy as np
import random
import torch
from glob import glob
import os
import json
from sklearn.preprocessing import StandardScaler

class Features:
    """
    The object containing all information of the selected features.

    Parameters
    ----------
    wav_dir : String
        The path to the root directory of the waveforms
    root_dir : String
        The path to the root directory of the features
    model_name : String
        The name of the model to get the features from
    layer : int
        The number of the layer to get the features from
    wav_format : String
        The format of the audio files.
        default: '.flac'
    num_files : int
        The number of features (utterances) to sample
        default: -1
    frames_per_ms : int
        The number of frames a model processes per millisecond
        default: 20
    """

    def __init__(
        self, wav_dir, root_dir, model_name, layer, wav_format='.flac', num_files=-1, frames_per_ms=20
    ):
        self.wav_dir = wav_dir
        self.root_dir = root_dir
        self.model_name = model_name
        self.layer = layer
        self.wav_format = wav_format
        self.num_files = num_files
        self.frames_per_ms = frames_per_ms

    def sample_features(self, speaker=None):
        """
        Randomly samples features (and their waveforms) from the specified model and returns the file paths as a list.

        Parameters
        ----------
        self : Class
            The object containing all information of the selected features
        speaker : String
            The code of the speaker to sample features from (used for mostly for the Buckeye corpus)

        Return
        ------
        features_sample : list, list
            List of file paths to the sampled features and their corresponding waveforms
        """

        if self.layer != -1:
            layer = 'layer_' + str(self.layer)
            if speaker is not None:
                all_features = sorted(glob(os.path.join(self.root_dir, self.model_name, layer, f'**/{speaker}*.npy'), recursive=True))
            else:
                all_features = sorted(glob(os.path.join(self.root_dir, self.model_name, layer, "**/*.npy"), recursive=True))
        else:
            if speaker is not None:
                all_features = sorted(glob(os.path.join(self.root_dir, self.model_name, f'**/{speaker}*.npy'), recursive=True))
            else:
                all_features = sorted(glob(os.path.join(self.root_dir, self.model_name, "**/*.npy"), recursive=True))

        if speaker is not None:
            all_wavs = sorted(glob(os.path.join(self.wav_dir, f'**/{speaker}*' + self.wav_format), recursive=True))
        else:
            all_wavs = sorted(glob(os.path.join(self.wav_dir, "**/*" + self.wav_format), recursive=True))

        if self.num_files == -1: # sample all the data
            return all_features, all_wavs
        
        paired_sample = list(zip(all_features, all_wavs))
        sample = random.sample(paired_sample, self.num_files)
        feature_sample, wavs_sample = zip(*sample)
        return feature_sample, wavs_sample

    def load_features(self, files):
        """
        Load the sampled features from file paths

        Parameters
        ----------
        self : Class
            The object containing all information of the selected features
        files : list (String)
            List of file paths to the sampled features

        Return
        ------
        features : list
            A list of features loaded from the file paths
        """

        features = []

        for file in files:
            feature = torch.from_numpy(np.load(file))
            if len(feature.shape) == 1: # if only one dimension, add a dimension
                features.append(feature.unsqueeze(0))
            else:
                features.append(feature)
        return features
    
    def normalize_features(self, features):
        """
        Normalizes the features to have a mean of 0 and a standard deviation of 1

        Parameters
        ----------
        self : Class
            The object containing all information of the selected features
        features : numpy.ndarray
            The features to normalize

        Returns
        -------
        normalized_features : numpy.ndarray
            The normalized features
        """

        stacked_features = torch.cat(features, dim=0) # concatenate all features into one tensor with size (sum_seq_len, feature_dim (channels))

        scaler = StandardScaler()
        scaler.partial_fit(stacked_features) # (n_samples, n_features)
        normalized_features = []
        for feature in features:
            normalized_features.append(torch.from_numpy(scaler.transform(feature))) # (n_samples, n_features)
        return normalized_features

    def get_speakers(self, speaker_file):
        """
        Get a list of speakers based on the dataset split.

        Parameters
        ----------
        self : Class
            The object containing all information of the selected features

        Return
        ------
        speaker_list : list (String)
            A list of speakers in the dataset
        """

        if os.path.splitext(speaker_file)[1] == ".json":
            language = os.path.split(self.wav_dir)[1]
            with open(speaker_file) as f:
                metadata = json.load(f)
                speaker_list = metadata['subsets'][f'{language}']['items']['wav_list']['files_list']
            speakers = []
            for speaker_file in speaker_list:
                speakers.append(os.path.splitext(os.path.split(speaker_file)[1])[0])
        else:
            speakers = []
            with open(speaker_file) as f:
                for line in f:
                    speakers.append(line.strip())
        
        return sorted(speakers)

    def get_frame_num(self, seconds):
        """
        Convert seconds to feature frame number

        Parameters
        ----------
        self : Class
            The object containing all information of the selected features
        seconds : float or ndarray (float)
            The number of seconds (of audio) to convert to frames

        Return
        ------
        output : int
            The feature frame number corresponding to the given number of seconds 
        """

        return np.round(seconds / self.frames_per_ms * 1000) # seconds (= samples / sample_rate) / x ms per frame * 1000ms per second

    def get_sample_second(self, frame_num):
        """
        Convert feature frame number to seconds

        Parameters
        ----------
        self : Class
            The object containing all information of the selected features
        frame_num : float
            The frame number (of features) to convert to seconds

        Return
        ------
        output : double
            The number of seconds corresponding to the given feature frame number
        """

        return frame_num * self.frames_per_ms / 1000 # frame_num * x ms per frame / 1000ms per second
"""
Encode audio files to MFCC features and save these embeddings.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os

import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as T

class EncodeAudio:
    def __init__(
        self, model_name, data_dir, save_dir, extension
    ):
        self.model_name = model_name
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.extension = extension

    def save_embedding(self, wav, file_path):
        """
        Push the audio through the model and save the embeddings to the save directory.
        """
        
        if self.model_name == "mfcc":
            self.encode_mfcc(wav, file_path)

    def preemphasis(self, signal, coeff=0.97):
        """
        Perform preemphasis on the signal.

        Parameters
        ----------
        signal : tensor
            The audio waveform
        coeff : float
            The preemphasis coefficient
            default: 0.97

        Return
        ------
        output : tensor
            The preemphasized signal
        """

        return torch.cat((signal[0][0].unsqueeze(0), (signal[0][1:] - coeff*signal[0][:-1]))).unsqueeze(0)
    
    def encode_mfcc(self, wav, file_path):
        """
        Determines the embeddings of the audio using MFCCs.
        Saves the embeddings to the save directory using the same structure as the dataset directory.

        Parameters
        ----------
        self : encoder object
            The model type, data directory, save directory, and extension.
        wav : tensor
            The audio waveform
        sr : int
            The sample rate of the audio
        file_path : String
            The path to the audio file

        Return
        ------
        output : N/A
        """

        if wav.shape[-1] < 200:
            wav = F.pad(wav, (200 - wav.shape[-1], 200 - wav.shape[-1]))

        wav = self.preemphasis(wav, coeff=0.97)

        f_s = 16000
        n_fft = int(np.floor(0.025*f_s)) #25ms window length (make 20ms for speech models)
        stride = int(np.floor(0.01*f_s)) #10ms (make 20.1ms for same number of frames as the speech models)
        transform = T.MFCC(sample_rate=f_s, n_mfcc=13, melkwargs={
        "n_fft": n_fft,
        "n_mels": 24,
        "hop_length": stride,
        "f_min": 64,
        "f_max": 8000,
        },)
        mfcc = transform(wav)
        mfcc = mfcc.permute(0, 2, 1)

        _, last_dir = os.path.split(self.save_dir)
        parts = str(file_path).split('/')
        copy_index = parts.index(last_dir)
        path_suffix = '/'.join(parts[copy_index + 1:])

        out_path = (self.save_dir.joinpath(f'{self.model_name}')) / path_suffix
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), mfcc.squeeze().cpu().numpy())

    def get_encodings(self):
        """
        Return the encodings for the dataset.
        """

        # walk through data directory
        for (dirpath, _, filenames) in os.walk(self.data_dir):
            if not filenames: # no files in directory
                continue
            
            # not in root of dataset path
            if dirpath is not self.data_dir:
                sub_dir = dirpath.split("/")[-1]
                print('subdir', sub_dir)

                # walk through files in directory
                for file in tqdm(filenames):
                    if not file.endswith(self.extension): # ensure only audio files are processed
                        continue

                    file_path = os.path.join(dirpath, file)
                    wav, sr = torchaudio.load(file_path, backend='soundfile')
                    assert sr == 16000
                    
                    self.save_embedding(wav, Path(file_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="model used to encode audio.",
        choices=["mfcc"],
        default="mfcc",
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    args = parser.parse_args() # python3 utils/encode.py mfcc /media/hdd/data/librispeech/dev-clean/ /media/hdd/embeddings/librispeech --extension=.flac
                               # python3 utils/encode.py mfcc /media/hdd/data/buckeye_segments/dev/ /media/hdd/embeddings/buckeye/dev --extension=.wav

    encoder = EncodeAudio(args.model, args.in_dir, args.out_dir, args.extension)
    encoder.get_encodings()
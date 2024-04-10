"""
Main script to segment audio using ES-KMeans, and evaluate the resulting segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

from utils import data_process
from wordseg import landmark_seg

# ~~~~~~~~~~ Sample a audio file and its alignments ~~~~~~~~~~

data = data_process.Features(wav_dir='/media/hdd/data/librispeech/dev-clean', root_dir='/media/hdd/embeddings/librispeech/', model_name='mfcc', layer=-1, data_dir='/media/hdd/data/librispeech_alignments/', alignment_format='.TextGrid', num_files=1)

sample, wavs = data.sample_embeddings() # sample from the feature embeddings
embeddings = data.load_embeddings(sample) # load the sampled embeddings
norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings

index_del = []
for i, embedding in enumerate(norm_embeddings): # delete embeddings with only one frame
    if embedding.shape[0] == 1:
        index_del.append(i)

for i in sorted(index_del, reverse=True):
    del sample[i]
    del embeddings[i]
    del norm_embeddings[i]

if len(sample) == 0:
    print('No embeddings to segment, sampled a file with only one frame.')
    exit()

print('1. Sampling:', sample[0], wavs[0])
print(embeddings[0].shape)
print(norm_embeddings[0].shape)

# Get alignements
alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files
print(data.alignment_data[0].text, data.alignment_data[0].end)
print('\n')

# ~~~~~~~~~~ Get the landmarks for the sampled audio file ~~~~~~~~~~

print('2. Landmarks')
for wav in wavs:
    boudaries = landmark_seg.get_boundaries(wav, fs=16000)
    print(boudaries)
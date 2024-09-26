# Embedded Segmental K-Means Plus (ES-KMeans+)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.14486)

This is the repository for the updated and improved embedded segmental K-means (ES-KMeans) algorithm from the paper at [https://arxiv.org/abs/1703.08135](https://arxiv.org/abs/1703.08135). ES-KMeans+ updates the original algorithm with a batched loading approach with lightweight data structures. Better landmarks are used and self-supervised features with an updated embedding approach is implemented. This new approach is introduced in the linked paper.

## Preliminaries

**Datasets**

- [ZeroSpeech](https://download.zerospeech.com/) Challenge Corpus (Track 2).
- [LibriSpeech](https://www.openslr.org/12) Corpus (dev-clean split) with alignments found [here](https://zenodo.org/records/2619474).
- [Buckeye](https://buckeyecorpus.osu.edu/) Corpus with splits found [here](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits) and alignments found [here](https://github.com/kamperh/vqwordseg/releases/tag/v1.0).

**Pre-Process and Encode Speech Data**

Use VAD to extract utterances from long speech files (specifically for ZeroSpeech and BuckEye) by cloning and following the recipes in the repository at [https://github.com/s-malan/data-process](https://github.com/s-malan/data-process).

**Encode Utterances**

Use pre-trained speech models or signal processing methods to encode speech utterances. Example code can be found here [https://github.com/bshall/hubert/blob/main/encode.py](https://github.com/bshall/hubert/blob/main/encode.py) using HuBERT-base for self-supervised audio encoding.
Save the feature encodings as .npy files with the file path as: 

    model_name/layer_#/relative/path/to/input/audio.npy

where # is replaced with an integer of the self-supervised model layer used for encoding, and as:

    model_name/relative/path/to/input/audio.npy

when signal processing methods like MFCCs are used.

**Extract Possible Word Boundaries (Landmarks)**

Clone and follow the recipe of the unsupervised prominence-based word segmentation repository [https://github.com/s-malan/prom-word-seg](https://github.com/s-malan/prom-word-seg).
If (like in ES-KMeans) SylSeg is used to determine the landmarks the get_boundaries function in landmarks_seg.py can be used.

**ZeroSpeech Repository**

Clone the ZeroSpeech repository at [https://github.com/zerospeech/benchmarks](https://github.com/zerospeech/benchmarks) to use the ZeroSpeech toolkit used for benchmark resources and evaluation scripts.

## Example Usage

**ES-KMeans+**

    python3 eskmeans+.py model_name layer path/to/audio path/to/features path/to/landmarks path/to/output k_max --extension --sample_size --speaker

The naming and format conventions used requires the preliminary scripts to be followed.
The **layer** argument selects a specific model layer, use -1 if the model has no layers (such as MFCC features).
The **k_max** argument specifies the number of K-means clusters used.
The **extension** argument specifies the format of the audio files (.wav or the default .flac). 
The **sample_size** argument controls how many utterances are sampled, the default is -1 to sample all utterances. 
The **speaker** argument let's you supply a .json file with speaker names whereafter the script will run speaker specific clustering on all the provided speakers, the default of None selects all speakers in a speaker-independent setting.

**Evaluation**

To evaluate the resultant hypothesized word boundaries and cluster assignments, clone and follow the recipe of the evaluation repository at [https://github.com/s-malan/evaluation](https://github.com/s-malan/evaluation).

For the ZeroSpeech Challenge dataset, use the ZeroSpeech toolkit's built in evaluation script.

## Contributors

- Simon Malan
- [Benjamin van Niekerk](https://scholar.google.com/citations?user=zCokvy8AAAAJ&hl=en&oi=ao)
- [Herman Kamper](https://www.kamperh.com/)

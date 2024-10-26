"""
Main functions and class for eskmeans+ segmentation and clustering.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import random
import numpy as np
from numba import njit
from tqdm import tqdm
from wordseg import subsample
from sklearn.decomposition import PCA
import faiss
import itertools

class ESKmeans():
    """
    Embedded segmental K-means PLUS.

    Segmentation and clustering are carried out using this class.

    Parameters
    ----------
    n_slices_min : int
        The minimum number of landmarks over which an embedding can be calculated.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be calculated.
    min_duration : int
        Minimum duration of a segment.
    p_boundary_init : float
        See `Utterances`.

    Attributes
    ----------
    data : Features
        The data object, containing the features.
    pca : PCA
        The PCA object, if PCA is used otherwise None
    samples : list (str)
        The list of the sampled file paths to the features of the utterances.
    landmarks : list (list (int))
        The landmarks (in frames) of the utterances.
    segments : list (list (tuple (int, int)))
        The segments (tuple of neighbouring landmarks) of the utterances.
    lengths : list (int)
        The number of landmarks per utterance.
    duration : list (list (int))
        The duration (in frames) of each of the segments of each utterance. 
    vec_ids : list (int)
        The vector ids for the utterance.
    boundaries : list (list (bool))
        The active/inactive landmarks of each utterance.
    K_max : int
        Maximum number of K-means components.
    D : int
        Number of utterances.
    """

    def __init__(self, n_slices_min=0, n_slices_max=20, min_duration=0, p_boundary_init=1.0):
        self.n_slices_min = n_slices_min
        self.n_slices_max = n_slices_max
        self.min_duration = min_duration
        self.p_boundary_init = p_boundary_init
    
    def get_boundaries(self, length):
        """
        Initialize boundaries of an utterance randomly with probability p_boundary_init.

        Parameters
        ----------
        length : int
            The number of landmarks for an utterance.

        Return
        ------
        boundaries : list (bool)
            The active/inactive landmarks of an utterance.
        """

        while True: # initializing boundaries randomly with probability
            boundaries = (np.random.rand(length) < self.p_boundary_init)
            boundaries[length - 1] = True

            # Test that `n_slices_max` is not exceeded
            indices = self.get_segmented_landmark_indices(boundaries=boundaries, lengths=length)
            if ((np.max([j[1] - j[0] for j in indices]) <= self.n_slices_max and
                    np.min([j[1] - j[0] for j in indices]) >= self.n_slices_min) or
                    (length <= self.n_slices_min)):
                break

        return boundaries
    
    def set_data(self, data, samples, landmarks, segments, lengths, durations, vec_ids, boundaries, K_max, pca=None):
        """
        Adds information to the class after it initialized.

        Parameters
        ----------
        data : Features
            The data object, containing the features.
        samples : list (str)
            The list of the sampled file paths to the features of the utterances.
        landmarks : list (list (int))
            The landmarks (in frames) of the utterances.
        segments : list (list (tuple (int, int)))
            The segments (tuple of neighbouring landmarks) of the utterances.
        lengths : list (int)
            The number of landmarks per utterance.
        duration : list (list (int))
            The duration (in frames) of each of the segments of each utterance. 
        vec_ids : list (int)
            The vector ids for the utterance.
        boundaries : list (bool)
            The active/inactive landmarks of each utterance.
        K_max : int
            Maximum number of K-means components.
        pca : PCA
            The PCA object, if PCA is used otherwise None
        """

        self.data = data
        self.pca = pca
        self.samples = samples
        self.landmarks = landmarks
        self.segments = segments
        self.lengths = lengths
        self.durations = durations
        self.vec_ids = vec_ids
        self.boundaries = boundaries
        self.K_max = K_max
        self.D = len(self.lengths)

    def get_segmented_embeds_i(self, i):
        """
        Return a list of embedding IDs according to the current segmentation
        for utterance `i`.

        Parameters
        ----------
        i : int
            The index of the utterance.

        Return
        ------
        embed_ids : list (int)
            The list of embedding IDs for utterance i.
        """

        embed_ids = []
        j_prev = 0
        for j in range(self.lengths[i]):
            if self.boundaries[i][j]:
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in vec_ids
                k += j_prev  # this is the index of the seq[j_prev:j] in vec_ids
                embed_ids.append(self.vec_ids[i][k])
                j_prev = j + 1
        return embed_ids
    
    def get_segmented_landmark_indices(self, boundaries, lengths):
        """
        Return a list of tuple, where every tuple is the start (inclusive) and
        end (exclusive) landmark index for the segmented embeddings.

        Parameters
        ----------
        boundaries : list (bool)
            The active/inactive landmarks of an utterance.
        lengths : int
            The number of landmarks for an utterance.

        Return
        ------
        indices : list (tuple)
            The list of tuples of the start and end landmarks for the segmented embeddings.
        """

        indices = []
        j_prev = 0
        for j in np.where(boundaries[:lengths])[0]:
            indices.append((j_prev, j + 1))
            j_prev = j + 1

        return indices
    
    def get_vec_embed_neg_len_sqrd_norms(self, embeddings, vec_ids, durations):
        """
        Calculates the negative squared L2 distance of the embeddings to the closest cluster centroid.
        Also returns the indices of the closest centroid.

        Parameters
        ----------
        embeddings : numpy.ndarray (utterance_length, embedding_dim)
            The embeddings of the utterance.
        vec_ids : list (int)
            The vector ids for the utterance.
        durations : list (int)
            The duration of each segment of the utterance.

        Return
        ------
        vec_embed_neg_len_sqrd_norms : numpy.ndarray (float)
            The duration weighted negative squared L2 distance of the embeddings to the closest cluster centroid.
        indices : numpy.ndarray (int)
            The cluster assignment for each embedding in an utterance.
        """

        # Get scores
        vec_embed_neg_len_sqrd_norms = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            
            # find negative L2 distance to closest centroid
            dist, _ = self.acoustic_model.index.search(embeddings[embed_id,:].reshape(1,embeddings.shape[-1]), 1)
            vec_embed_neg_len_sqrd_norms[i] = -1*dist # negative squared L2 distance

            # Scale log marginals by number of frames
            if durations[i] == -1:
                vec_embed_neg_len_sqrd_norms[i] = -np.inf
            else:
                vec_embed_neg_len_sqrd_norms[i] *= durations[i]

        return vec_embed_neg_len_sqrd_norms
    
    def segment_utt_i(self, i, extract=False):
        """
        Segment new boundaries and cluster new segments for utterance `i`.

        Parameters
        ----------
        i : int
            The index of the utterance used in the cluster assignment.
        extract : bool
            If True, the classes are extracted and returned.
            Else the new segmentation is found and set.

        Return
        ------
        sum_neg_len_sqrd_norm : float
            The sum of the scores in vec_embed_neg_len_sqrd_norms for the embeddings for the final segmentation.
        classes : list (int)
            The classes assigned to each segment of utterance `i`.
        """

        # sample the current utterance and downsample all possible segmentations
        embeddings = self.data.load_features([self.samples[i]])[0]

        if extract: # active segments
            active_landm = [self.landmarks[i][j] for j in range(len(self.landmarks[i])) if self.boundaries[i][j]]
            active_landm.insert(0, 0)
            all_segments = list(itertools.pairwise(active_landm))
        else:
            # get all possible segments:
            all_segments = self.segments[i]

        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
            downsampled_embeddings = []
            for a, b in all_segments:
                if b > embeddings.shape[0]: # if the segment is longer than the utterance, stop at last frame
                    if a < embeddings.shape[0]:
                        downsampled_embeddings.append(embeddings[a:, :].mean(0))
                    elif a == embeddings.shape[0]:
                        downsampled_embeddings.append(embeddings[a-1:, :].mean(0))
                elif a == b: # if the segment is empty, add one frame
                    downsampled_embeddings.append(embeddings[a-1:b+1, :].mean(0))
                else:
                    downsampled_embeddings.append(embeddings[a:b, :].mean(0))

            embeddings = np.stack(downsampled_embeddings)
            del downsampled_embeddings
        else:
            embeddings = subsample.downsample([embeddings], [all_segments], n=10)

        # normalize embedding
        downsampled_utterance = []
        for frame_i in range(embeddings.shape[0]):
            cur_embed = embeddings[frame_i, :]
            norm = np.linalg.norm(cur_embed)
            downsampled_utterance.append(cur_embed / np.linalg.norm(cur_embed))
            assert norm != 0.
        embeddings = np.stack(downsampled_utterance)

        if extract: # assign each segment to the closest cluster centroid
            classes = [-1]*len(all_segments)
            for i_index in range(len(classes)):
                _, index = self.acoustic_model.index.search(embeddings[i_index].reshape(1, embeddings.shape[-1]), 1)
                classes[i_index] = index[0][0]
            return _, classes

        vec_embed_neg_len_sqrd_norms = self.get_vec_embed_neg_len_sqrd_norms(embeddings, # all embeddings
        self.vec_ids[i], self.durations[i])

        sum_neg_len_sqrd_norm, self.boundaries[i] = forward_backward_kmeans_viterbi(
        vec_embed_neg_len_sqrd_norms, self.lengths[i], self.n_slices_min, self.n_slices_max)

        return sum_neg_len_sqrd_norm, None

    def segment(self, n_iterations):
        """
        Segment the embedded utterances using ES-KMeans+.

        Parameters
        ----------
        n_iterations : int
            The number of iterations for the k-means+ algorithm

        Return
        ------
        sum_neg_len_sqrd_norm : float
            The sum of the scores in vec_embed_neg_len_sqrd_norms for the embeddings for the final segmentation.
        classes : list (list (int))
            The classes assigned to each segment of all utterances.
        """

        for i_iter in tqdm(range(n_iterations+1), desc="Iteration"):
            # Cluster:
            embeddings = []
            for i_utt, sample in enumerate(self.samples): # cluster active segments
                active_landm = [self.landmarks[i_utt][j] for j in range(len(self.landmarks[i_utt])) if self.boundaries[i_utt][j]]
                active_landm.insert(0, 0)
                segment = list(itertools.pairwise(active_landm))

                downsampled_embedding = []
                if self.pca is not None:
                    embedding = self.pca.transform(self.data.load_features([sample])[0])
                    for a, b in segment:
                        if b > embedding.shape[0]: # if the segment is longer than the utterance, stop at last frame
                            if a < embedding.shape[0]:
                                downsampled_embedding.append(embedding[a:, :].mean(0))
                            elif a == embedding.shape[0]:
                                downsampled_embedding.append(embedding[a-1:, :].mean(0))
                        elif a == b: # if the segment is empty, add one frame
                            downsampled_embedding.append(embedding[a-1:b+1, :].mean(0))
                        else:
                            downsampled_embedding.append(embedding[a:b, :].mean(0))
                    embeddings.append(np.stack(downsampled_embedding))
                else:
                    embedding = self.data.load_features([sample])[0]
                    embeddings.append(subsample.downsample([embedding], [segment], n=10))   

            # Normalize embedding (frame-wise unit sphere)
            for i_utt in range(len(embeddings)):
                downsampled_utterance = []
                for i in range(embeddings[i_utt].shape[0]):
                    cur_embed = embeddings[i_utt][i, :]
                    norm = np.linalg.norm(cur_embed)
                    downsampled_utterance.append(cur_embed / np.linalg.norm(cur_embed))
                    assert norm != 0.
                embeddings[i_utt] = np.stack(downsampled_utterance)
            
            # Cluster segment embeddings
            embeddings = np.concatenate(embeddings, axis=0)
            self.acoustic_model = faiss.Kmeans(embeddings.shape[1], self.K_max, niter=15, nredo=3, verbose=True)
            self.acoustic_model.train(embeddings)

            # Segment or extract classes and return:
            classes = [None]*self.D
            sum_neg_len_sqrd_norm = 0
            utt_order = list(range(self.D))
            random.shuffle(utt_order)

            for i_utt in tqdm(utt_order, desc="Segmenting Utterances"):
                if i_iter == n_iterations:
                    _, i_classes = self.segment_utt_i(i_utt, extract=True)
                    classes[i_utt] = i_classes
                else:
                    sum_neg_len_sqrd_norm_utt, _ = self.segment_utt_i(i_utt)
                    sum_neg_len_sqrd_norm += sum_neg_len_sqrd_norm_utt
        
        return sum_neg_len_sqrd_norm, classes

@njit
def forward_backward_kmeans_viterbi(vec_embed_neg_len_sqrd_norms, N, n_slices_min, n_slices_max):
    """
    Segmental K-means viterbi segmentation of an utterance of length `N` based
    on its `vec_embed_neg_len_sqrd_norms` vector.

    Parameters
    ----------
    vec_embed_neg_len_sqrd_norms : numpy.ndarray (float)
            The duration weighted negative squared L2 distance of the embeddings to the closest cluster centroid.
    all_classes : numpy.ndarray (int)
            The cluster assignment for each embedding in an utterance.
    N : int
        The number of landmarks for an utterance.
    n_slices_min : int
        The minimum number of landmarks over which an embedding can be calculated.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be calculated.

    Return
    ------
    sum_neg_len_sqrd_norm : float
        The sum of the scores in vec_embed_neg_len_sqrd_norms for the embeddings for the final segmentation.
    boundaries : list (bool)
            The active/inactive landmarks of an utterance.
    classes : list ((int)
        The classes assigned to each segment of the utterance.
    """

    n_slices_min_cut = -(n_slices_min - 1) if n_slices_min > 1 else None

    boundaries = np.zeros(N, dtype=np.bool_)
    boundaries[-1] = True
    gammas = np.ones(N)
    gammas[0] = 0.0

    # Forward filtering
    i = 0
    for t in range(1, N):
        if np.all(vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] +
                gammas[:t][-n_slices_max:] == -np.inf):
            gammas[t] = -np.inf
        elif n_slices_min_cut is not None:
            gammas[t] = np.max(
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
                gammas[:t][-n_slices_max:n_slices_min_cut]
                )
        else: # to make numba work
            gammas[t] = np.max(
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] +
                gammas[:t][-n_slices_max:]
                )
        i += t
    
    # Backward segmentation
    t = N
    sum_neg_len_sqrd_norm = 0.
    while True:
        i = int(0.5*(t - 1)*t)
        if n_slices_min_cut is not None:
            q_t = (
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
                gammas[:t][-n_slices_max:n_slices_min_cut]
                )
        else: # to make numba work
            q_t = (
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] +
                gammas[:t][-n_slices_max:]
                )

        if np.all(q_t == -np.inf):

            # Look for first point where we can actually sample and insert a boundary at this point
            while np.all(q_t == -np.inf):
                t = t - 1
                if t == 0:
                    break
                i = int(0.5*(t - 1)*t)
                q_t = (vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] + gammas[:t][-n_slices_max:])

            boundaries[t - 1] = True  # insert the boundary

        q_t = q_t[::-1]
        k = np.argmax(q_t) + 1
        if n_slices_min_cut is not None:
            k += n_slices_min - 1
        
        sum_neg_len_sqrd_norm += vec_embed_neg_len_sqrd_norms[i + t - k]
        if t - k - 1 < 0:
            break
        boundaries[t - k - 1] = True
        t = t - k

    return sum_neg_len_sqrd_norm, boundaries
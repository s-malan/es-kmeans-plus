"""
Main functions and class for eskmeans segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import numpy as np
import random
from cluster import KMeans_Herman
import timeit

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import data_process
import landmark_seg, downsample

class ESKmeans_Herman():
    """
    Embedded segmental K-means.

    Segmentation and clustering are carried out using this class. Variables
    related to the segmentation are stored in the `utterances` attribute, which
    deals with all utterance-level information but knows nothing about the
    acoustics. The `kmeans` attribute deals with all the acoustic embedding
    operations. In member functions, index `i` generally refers to the index of
    an utterance.

    Parameters
    ----------
    K_max : int
        Maximum number of components.
    embedding_mats : dict of matrix
        The matrices of embeddings for every utterance.
    vec_ids_dict : dict of vector of int
        For every utterance, the vector IDs (see `Utterances`).
    landmarks_dict : dict of list of int
        For every utterance, the landmark points at which word boundaries are
        considered, given in the number of frames (10 ms units) from the start
        of each utterance. There is an implicit landmark at the start of every
        utterance.
    durations_dict : dict of vector of int
        The shape of this dict is the same as that of `vec_ids_dict`, but here
        the duration (in frames) of each of the embeddings are given.
    n_slices_min : int
        The minimum number of landmarks over which an embedding can be
        calculated.
    n_slices_max : int
        The maximum number of landmarks over which an embedding can be
        calculated.
    min_duration : int
        Minimum duration of a segment.
    wip : float
        Word insertion penalty.
    p_boundary_init : float
        See `Utterances`.
    init_assignments : str
        This setting determines how the initial acoustic model assignments are
        determined: "rand" assigns data vectors randomly; "each-in-own" assigns
        each data point to a component of its own; and "spread" makes an
        attempt to spread data vectors evenly over the components.

    Attributes
    ----------
    utterances : Utterances
        Knows nothing about the acoustics. The indices in the `vec_ids`
        attribute refers to the embedding at the corresponding row in
        `acoustic_model.X`.
    acoustic_model : KMeans
        Knows nothing about utterance-level information. All embeddings are
        stored in this class in its `X` attribute.
    ids_to_utterance_labels : list of str
        Keeps track of utterance labels for a specific utterance ID.
    """

    def __init__(self, K_max, embeddings, vec_ids, durations,
            landmarks, lengths, n_slices_min=0, n_slices_max=20, min_duration=0,
            p_boundary_init=0.5, wip=0):

        # Attributes from parameters
        self.durations = durations
        self.landmarks = landmarks
        self.lengths = lengths
        self.n_slices_min = n_slices_min # !! Benji does not use this
        self.n_slices_max = n_slices_max
        self.wip = wip

        # Process embeddings into a single matrix, and vec_ids into a list (entry for each utterance)
        self.embeddings, vec_ids, _ = process_embeddings(embeddings, vec_ids) # !! only do if more than one utterance at a time

        # Set up utterance information
        self.N = self.embeddings.shape[0]
        self.D = len(self.lengths) # number of utterances
        self.max_num_landmarks = max(self.lengths)

        # Vec IDs
        self.vec_ids = -1*np.ones(
            (self.D, int(self.max_num_landmarks*(self.max_num_landmarks + 1)/2)), dtype=np.int32)
        for i_vec_id, vec_id in enumerate(vec_ids):
            self.vec_ids[i_vec_id, :len(vec_id)] = vec_id

        # Durations
        self.durations = -1*np.ones((self.D, int(self.max_num_landmarks*(self.max_num_landmarks + 1)/2)), dtype=np.int32)
        for i_duration_vec, duration_vec in enumerate(durations):
            if not (min_duration == 0 or len(duration_vec) == 1):
                cur_duration_vec = np.array(duration_vec, dtype=np.float64)
                # cur_duration_vec[cur_duration_vec < min_duration] = -np.nan
                cur_duration_vec[cur_duration_vec < min_duration] = -1
                # if np.all(np.isnan(cur_duration_vec)):
                if np.all(cur_duration_vec == -1):
                    cur_duration_vec[np.argmax(duration_vec)] = np.max(duration_vec)
                duration_vec = cur_duration_vec
            self.durations[i_duration_vec, :len(duration_vec)] = duration_vec
        
        # Boundaries
        self.boundaries = np.zeros((self.D, self.max_num_landmarks), dtype=bool)
        for i in range(self.D):
            N_i = self.lengths[i]
            while True: # initializing boundaries randomly with probability
                self.boundaries[i, 0:N_i] = (np.random.rand(N_i) < p_boundary_init)
                self.boundaries[i, N_i - 1] = True

                # Don't allow all disregarded embeddings for initialization
                if np.all(np.asarray(self.get_segmented_embeds_i(i)) == -1):
                    continue

                # Test that `n_slices_max` is not exceeded
                indices = self.get_segmented_landmark_indices(i)
                if ((np.max([j[1] - j[0] for j in indices]) <= n_slices_max and
                        np.min([j[1] - j[0] for j in indices]) >= n_slices_min) or
                        (N_i <= n_slices_min)):
                    break

        # Initialize the K-means components !! hardcoded to be spread
        init_embeds = []
        for i in range(self.D):
            init_embeds.extend(self.get_segmented_embeds_i(i))
        init_embeds = np.array(init_embeds, dtype=int)
        init_embeds = init_embeds[np.where(init_embeds != -1)]

        assignments = -1*np.ones(self.N, dtype=int) # TODO make one-hot
        n_init_embeds = len(init_embeds)
        assignment_list = (list(range(K_max))*int(np.ceil(float(n_init_embeds)/K_max)))[:n_init_embeds]
        random.shuffle(assignment_list)
        assignments[init_embeds] = np.array(assignment_list)

        self.acoustic_model = KMeans_Herman(self.embeddings, K_max, assignments) # TODO make sure embeddings are in correct format (N utterance segments, D features after downsampled)
        print('ESKMEANS initial assignments:', self.acoustic_model.assignments, self.boundaries)
    
    def get_segmented_embeds_i(self, i):
        """
        Return a list of embedding IDs according to the current segmentation
        for utterance `i`.
        """
        embed_ids = []
        j_prev = 0
        for j in range(self.lengths[i]):
            if self.boundaries[i, j]:
                # We aim to extract seq[j_prev:j+1]. Let the location of this
                # ID be `vec_ids[i, k]`, and we need to find k.
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in `vec_ids[i]`
                k += j_prev  # this is the index of the seq[j_prev:j] in `vec_ids[i]`
                embed_ids.append(self.vec_ids[i, k])
                j_prev = j + 1
        return embed_ids
    
    def get_segmented_landmark_indices(self, i):
        """
        Return a list of tuple, where every tuple is the start (inclusive) and
        end (exclusive) landmark index for the segmented embeddings.
        """
        indices = []
        j_prev = 0
        for j in np.where(self.boundaries[i][:self.lengths[i]])[0]:
            indices.append((j_prev, j + 1))
            j_prev = j + 1
        return indices
    
    def get_vec_embed_neg_len_sqrd_norms(self, vec_ids, durations):

        # Get scores
        vec_embed_neg_len_sqrd_norms = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            vec_embed_neg_len_sqrd_norms[i] = self.acoustic_model.max_neg_sqrd_norm_i(embed_id)

            # Scale log marginals by number of frames
            # if np.isnan(durations[i]):
            if durations[i] == -1:
                vec_embed_neg_len_sqrd_norms[i] = -np.inf
            else:
                vec_embed_neg_len_sqrd_norms[i] *= durations[i]#**self.time_power_term

        return vec_embed_neg_len_sqrd_norms + self.wip
    
    def get_max_unsup_transcript_i(self, i):
        """
        Return a list of the best components for current segmentation of `i`.
        """
        return self.acoustic_model.get_max_assignments(self.get_segmented_embeds_i(i))
    
    def get_unsup_transcript_i(self, i):
        """
        Return a list of the current component assignments for the current
        segmentation of `i`.
        """
        return list(self.acoustic_model.get_assignments(self.get_segmented_embeds_i(i)))
    
    def segment_utt_i(self, i, old_k):
        """
        Segment new boundaries and cluster new segments for utterance `i`.

        Return
        ------
        sum_neg_len_sqrd_norm : float
            The length-weighted K-means objective for this utterance.
        """

        old_embeds = self.get_segmented_embeds_i(i)
        print('old', old_embeds, old_k, self.boundaries[i])

        N = self.lengths[i]
        vec_embed_neg_len_sqrd_norms = self.get_vec_embed_neg_len_sqrd_norms( # get the score of each segment of the utterance
            self.vec_ids[i, :int((N**2 + N)/2)],
            self.durations[i, :int((N**2 + N)/2)]
            )

        # Get new boundaries
        sum_neg_len_sqrd_norm, self.boundaries[i, :N] = forward_backward_kmeans_viterbi(
        vec_embed_neg_len_sqrd_norms, N, self.n_slices_min, self.n_slices_max)
        print('new boundaries after viterbi', self.boundaries[i])

        # Remove old embeddings and add new ones; this is equivalent to
        # assigning the new embeddings and updating the means.
        new_embeds = self.get_segmented_embeds_i(i)
        new_k = self.get_max_unsup_transcript_i(i)
        print('new', new_embeds, new_k)

        for i_embed in old_embeds:
            if i_embed == -1:
                continue  # don't remove a non-embedding (would accidently remove the last embedding)
            self.acoustic_model.del_item(i_embed) # TODO below comments only delete if not in new_embeds
            print('del', i_embed)
        for i_embed, k in zip(new_embeds, new_k):
            print('new assigment', i_embed, k)
            self.acoustic_model.add_item(i_embed, k)
        self.acoustic_model.clean_components()

        # # only update if changes were made
        # del_embeds = []
        # add_embeds = []
        # add_k = []
        # for embed, k in zip(old_embeds, old_k):
        #     if not (embed in new_embeds and k == new_k[new_embeds.index(embed)]):
        #         del_embeds.append(embed)
        
        # if len(del_embeds) > 0:
        #     for embed, k in zip(new_embeds, new_k):
        #         if not (embed in old_embeds and k == old_k[old_embeds.index(embed)]):
        #             add_embeds.append(embed)
        #             add_k.append(k)
        
        # for i_embed in del_embeds:
        #     if i_embed == -1:
        #         continue  # don't remove a non-embedding (would accidently remove the last embedding)
        #     self.acoustic_model.del_item(i_embed) # TODO only delete if not in new_embeds
        #     print('del', i_embed)
        # for i_embed, k in zip(add_embeds, add_k):
        #     print('new assigment', i_embed, k)
        #     self.acoustic_model.add_item(i_embed, k)
        # self.acoustic_model.clean_components()

        # print(self.acoustic_model.assignments)
        return sum_neg_len_sqrd_norm, new_k

    def segment(self, n_iterations):
        """
        Segment the downsampled utterance using ES-KMeans.

        Parameters
        ----------
        # downsampled_utterances : list (tensor)
        #     The downsampled embeddings of the encoded utterances
        # segments : list of lists (of tuples)
        #     The segments (begin, end) to downsample for each utterance
        # k_max : int
        #     The maximum number of clusters
        n_iterations : int
            The number of iterations for the k-means algorithm

        Return
        ------
        segmentations : list (list)
            The segmentations of the downsampled utterances
        """

        old_k = [None]*self.D
        print(old_k)
        for iteration in range(n_iterations):
            print(f'\t~~~~~~~~~~~~~~~ Iteration {iteration+1} ~~~~~~~~~~~~~~~')

            sum_neg_len_sqrd_norm = 0
            utt_order = list(range(self.D))
            for i_utt in utt_order:
                if old_k[i_utt] is None:
                    old_k[i_utt] = self.get_max_unsup_transcript_i(i_utt)
                    print('old_k', old_k[i_utt])

                print(f'----- Utterance {i_utt+1} -----')
                sum_neg_len_sqrd_norm_utt, old_k[i_utt] = self.segment_utt_i(i_utt, old_k[i_utt])
                sum_neg_len_sqrd_norm += sum_neg_len_sqrd_norm_utt

            print('Sum of negative squared norm:', sum_neg_len_sqrd_norm)

        return sum_neg_len_sqrd_norm

def process_embeddings(embedding_mats, vec_ids_dict):
    """
    Process the embeddings and vector IDs into single data structures.

    Return
    ------
    (embeddings, vec_ids, utterance_labels_to_ids) : 
            (matrix of float, list of vector of int, list of str)
        All the embeddings are returned in a single matrix, with a `vec_id`
        vector for every utterance and a list of str indicating which `vec_id`
        goes with which original utterance label.
    """

    embeddings = []
    vec_ids = []
    ids_to_utterance_labels = []
    i_embed = 0
    n_embed = 0

    # Loop over utterances
    for i_utt, utt in enumerate(embedding_mats):
        ids_to_utterance_labels.append(i_utt)
        cur_vec_ids = np.array(vec_ids_dict[i_utt])

        # Loop over rows
        for i_row, row in enumerate(embedding_mats[i_utt]):
            n_embed += 1

            # Add it to the embeddings
            embeddings.append(row)

            # Update vec_ids_dict so that the index points to i_embed
            cur_vec_ids[np.where(np.array(vec_ids_dict[i_utt]) == i_row)[0]] = i_embed
            i_embed += 1

        # Add the updated entry in vec_ids_dict to the overall vec_ids list
        vec_ids.append(cur_vec_ids)

    # print('CHECK', embeddings, vec_ids, ids_to_utterance_labels)
    return (np.asarray(embeddings), vec_ids, ids_to_utterance_labels)

def forward_backward_kmeans_viterbi(vec_embed_neg_len_sqrd_norms, N,
        n_slices_min=0, n_slices_max=0):
    """
    Segmental K-means viterbi segmentation of an utterance of length `N` based
    on its `vec_embed_neg_len_sqrd_norms` vector and return a bool vector of
    boundaries.

    Parameters
    ----------
    vec_embed_neg_len_sqrd_norms : N(N + 1)/2 length vector
        For t = 1, 2, ..., N the entries `vec_embed_neg_len_sqrd_norms[i:i + t]`
        contains the log probabilties of sequence[0:t] up to sequence[t - 1:t],
        with i = t(t - 1)/2. If you have a NxN matrix where the upper
        triangular (i, j)'th entry is the log probability of sequence[i:j + 1],
        then by stacking the upper triangular terms column-wise, you get
        vec_embed_neg_len_sqrd_norms`. Written out:
        `vec_embed_neg_len_sqrd_norms` = [neg_len_sqrd_norm(seq[0:1]),
        neg_len_sqrd_norm(seq[0:2]), neg_len_sqrd_norm(seq[1:2]),
        neg_len_sqrd_norm(seq[0:3]), ..., neg_len_sqrd_norm(seq[N-1:N])].
    n_slices_max : int
        If 0, then the full length are considered. This won't necessarily lead
        to problems, since unassigned embeddings would still be ignored since
        their assignments are -1 and the would therefore have a log probability
        of -inf.
    i_utt : int
        If provided, index of the utterance for which to print a debug trace;
        this happens if it matches the global `i_debug_monitor`.

    Return
    ------
    (sum_neg_len_sqrd_norm, boundaries) : (float, vector of bool)
        The `sum_neg_len_sqrd_norm` is the sum of the scores in
        `vec_embed_neg_len_sqrd_norms` for the embeddings for the final
        segmentation.
    """

    n_slices_min_cut = -(n_slices_min - 1) if n_slices_min > 1 else None

    boundaries = np.zeros(N, dtype=bool)
    boundaries[-1] = True
    gammas = np.ones(N)
    gammas[0] = 0.0

    # Forward filtering
    i = 0
    for t in range(1, N):
        if np.all(vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:] +
                gammas[:t][-n_slices_max:] == -np.inf):
            gammas[t] = -np.inf
        else:
            gammas[t] = np.max(
                vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
                gammas[:t][-n_slices_max:n_slices_min_cut]
                )
        i += t

    # Backward segmentation
    t = N
    sum_neg_len_sqrd_norm = 0.
    while True:
        i = int(0.5*(t - 1)*t)
        q_t = (
            vec_embed_neg_len_sqrd_norms[i:i + t][-n_slices_max:n_slices_min_cut] +
            gammas[:t][-n_slices_max:n_slices_min_cut]
            )
        # assert not np.isnan(np.sum(q_t))
        if np.all(q_t == -np.inf):

            # Look for first point where we can actually sample and insert a boundary at this point
            while np.all(q_t == -np.inf):
                t = t - 1
                if t == 0:
                    break  # this is a very crappy utterance
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

def main():
    data = data_process.Features(wav_dir='/media/hdd/data/luna', root_dir='/media/hdd/embeddings/luna', model_name='mfcc', layer=-1, data_dir='/media/hdd/data/librispeech_alignments/', wav_format='.wav', alignment_format='.TextGrid', num_files=-1)

    sample, wavs = data.sample_embeddings() # sample from the feature embeddings
    embeddings = data.load_embeddings(sample) # load the sampled embeddings

    index_del = []
    for i, embedding in enumerate(embeddings): # delete embeddings with only one frame
        if embedding.shape[0] == 1:
            index_del.append(i)

    for i in sorted(index_del, reverse=True):
        del sample[i]
        del embeddings[i]
    
    n_slices_max = 6 # max number of landmarks a segment can span, n_landmarks_max in notebook
    landmarks = []
    segments = []
    lengths = []
    for wav in wavs: # for each utterance
        landmarks.append(np.ceil(landmark_seg.get_boundaries(wav, fs=16000)*100).astype(np.int32).tolist()[1:]) # get the boundaries
        segments.append(landmark_seg.get_segments(landmarks[-1], max_span = n_slices_max))
        lengths.append(len(landmarks[-1]))
    
    downsampled_utterances = downsample.downsample(embeddings, segments, n=10)

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
    
    for i_utt in range(len(downsampled_utterances)):
        print(len(vec_ids[i_utt]), len(segments[i_utt]), lengths[i_utt])
        print((vec_ids[i_utt]), (segments[i_utt]), lengths[i_utt])

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

    K_max = 5 # max number of clusters
    segmenter = ESKmeans_Herman(
        K_max, downsampled_utterances, vec_ids, durations, landmarks, lengths,
        p_boundary_init=0.5, n_slices_max=n_slices_max, min_duration=20
        )

    # Perform inference
    def runit():
        segmenter.segment(n_iterations=5)

    print('MY TIME:', timeit.timeit(runit, number=1))

    ground_truth_boundaries = [[10, 33, 62, 96, 139, 167],[52, 70, 120, 180],[23, 44, 77, 95, 109, 164, 191]] # based on landmarks
    # [10, 33, 62, 96, 139, 167] is [33, 62, 96, 139, 167] AND [52, 70, 120, 180] is [52, 70, 180] AND [23, 44, 77, 95, 109, 164, 191] is [23, 44, 95, 109, 191] if we do not consider silences
    # obtain clusters and landmarks
    for i in range(segmenter.D): # for each utterance
        print(segmenter.get_unsup_transcript_i(i), segmenter.get_segmented_embeds_i(i), segmenter.boundaries[i], landmarks[i], ground_truth_boundaries[i])
        print(landmarks[i]*segmenter.boundaries[i][0:len(landmarks[i])])

if __name__ == "__main__":
    main()
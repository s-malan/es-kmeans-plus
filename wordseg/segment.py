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
            landmarks, n_slices_min=0, n_slices_max=20, min_duration=0,
            p_boundary_init=0.5, init_assignments="rand", wip=0):

        # Attributes from parameters
        self.embeddings = embeddings
        self.N = embeddings.shape[0]
        self.vec_ids = vec_ids
        self.duration = durations
        self.landmarks = landmarks
        self.length = len(landmarks)
        self.n_slices_min = n_slices_min # !! Benji does not use this
        self.n_slices_max = n_slices_max
        self.wip = wip

        # Duration
        if not (min_duration == 0 or len(durations) == 1):
            cur_duration_vec = np.array(durations, dtype=np.float64)
            # cur_duration_vec[cur_duration_vec < min_duration] = -np.nan
            cur_duration_vec[cur_duration_vec < min_duration] = -1
            # if np.all(np.isnan(cur_duration_vec)):
            if np.all(cur_duration_vec == -1):
                cur_duration_vec[np.argmax(durations)] = np.max(durations)
            durations = cur_duration_vec
            self.duration = durations.astype(np.int32)
        
        # Boundaries:
        self.boundaries = np.zeros(self.length, dtype=bool)
        while True: # initializing boundaries randomly with probability
            self.boundaries[0:self.length] = (np.random.rand(self.length) < p_boundary_init)
            self.boundaries[self.length - 1] = True

            # Don't allow all disregarded embeddings for initialization
            if np.all(np.asarray(self.get_segmented_embeds_i()) == -1):
                continue

            # Test that `n_slices_max` is not exceeded
            indices = self.get_segmented_landmark_indices()
            if ((np.max([j[1] - j[0] for j in indices]) <= n_slices_max and
                    np.min([j[1] - j[0] for j in indices]) >= n_slices_min) or
                    (self.length <= n_slices_min)):
                break

        # Process embeddings into a single matrix, and vec_ids into a list (entry for each utterance)
        # embeddings, vec_ids, ids_to_utterance_labels = process_embeddings(embeddings, vec_ids) # !! only do if more than one utterance at a time
        # lengths = [len(landmarks[i]) for i in ids_to_utterance_labels]
        # landmarks = [landmarks[i] for i in ids_to_utterance_labels]
        # durations = [durations[i] for i in ids_to_utterance_labels]

        # Initialize the K-means components !! hardcoded to be random
        init_embeds = []
        init_embeds.extend(self.get_segmented_embeds_i())
        init_embeds = np.array(init_embeds, dtype=int)
        init_embeds = init_embeds[np.where(init_embeds != -1)]

        assignments = -1*np.ones(self.N, dtype=int)
        assignments[init_embeds] = np.random.randint(0, K_max, len(init_embeds))

        self.acoustic_model = KMeans_Herman(embeddings, K_max, assignments) # TODO make sure embeddings are in correct format (N utterance segments, D features after downsampled)
        print('ESKMEANS initial assignments:', self.acoustic_model.assignments, self.boundaries)
    
    def get_segmented_embeds_i(self):
        """
        Return a list of embedding IDs according to the current segmentation
        for utterance `i`.
        """
        embed_ids = []
        j_prev = 0
        for j in range(self.length):
            if self.boundaries[j]:
                # We aim to extract seq[j_prev:j+1]. Let the location of this
                # ID be `vec_ids[i, k]`, and we need to find k.
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in `vec_ids[i]`
                k += j_prev  # this is the index of the seq[j_prev:j] in `vec_ids[i]`
                embed_ids.append(self.vec_ids[k])
                j_prev = j + 1
        return embed_ids
    
    def get_segmented_landmark_indices(self):
        """
        Return a list of tuple, where every tuple is the start (inclusive) and
        end (exclusive) landmark index for the segmented embeddings.
        """
        indices = []
        j_prev = 0
        for j in np.where(self.boundaries[:self.length])[0]:
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
    
    def get_max_unsup_transcript_i(self):
        """
        Return a list of the best components for current segmentation of `i`.
        """
        return self.acoustic_model.get_max_assignments(self.get_segmented_embeds_i())
    
    def get_unsup_transcript_i(self):
        """
        Return a list of the current component assignments for the current
        segmentation of `i`.
        """
        return list(self.acoustic_model.get_assignments(self.get_segmented_embeds_i()))
    
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

        for iteration in range(n_iterations):
            print(f'~~~~~ Iteration {iteration+1} ~~~~~')
            old_embeds = self.get_segmented_embeds_i()
            print('old', old_embeds, self.boundaries, self.acoustic_model.assignments)

            vec_embed_neg_len_sqrd_norms = self.get_vec_embed_neg_len_sqrd_norms(self.vec_ids[:int((self.length**2 + self.length)/2)], self.duration[:int((self.length**2 + self.length)/2)])
            print(vec_embed_neg_len_sqrd_norms)

            # Get new boundaries
            sum_neg_len_sqrd_norm, self.boundaries[:self.length] = forward_backward_kmeans_viterbi( # TODO check if works
            vec_embed_neg_len_sqrd_norms, self.length, self.n_slices_min, self.n_slices_max)
            print('new boundaries after viterbi', self.boundaries)

            # Remove old embeddings and add new ones; this is equivalent to
            # assigning the new embeddings and updating the means.
            new_embeds = self.get_segmented_embeds_i()
            new_k = self.get_max_unsup_transcript_i()
            print('new', new_embeds, new_k)

            for i_embed in old_embeds:
                if i_embed == -1:
                    continue  # don't remove a non-embedding (would accidently remove the last embedding)
                self.acoustic_model.del_item(i_embed) # TODO only delete if not in new_embeds
                print('del', i_embed, self.acoustic_model.assignments)
            for i_embed, k in zip(new_embeds, new_k):
                print('new assigment', i_embed, k, self.acoustic_model.assignments)
                self.acoustic_model.add_item(i_embed, k)
            self.acoustic_model.clean_components()

            print(self.acoustic_model.assignments)

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
    for _, utt in enumerate(sorted(embedding_mats)):
        ids_to_utterance_labels.append(utt)
        cur_vec_ids = vec_ids_dict[utt].copy()

        # Loop over rows
        for i_row, row in enumerate(embedding_mats[utt]):
            n_embed += 1

            # Add it to the embeddings
            embeddings.append(row)

            # Update vec_ids_dict so that the index points to i_embed
            cur_vec_ids[np.where(vec_ids_dict[utt] == i_row)[0]] = i_embed
            i_embed += 1

        # Add the updated entry in vec_ids_dict to the overall vec_ids list
        vec_ids.append(cur_vec_ids)

    print('CHECK', vec_ids, embeddings, ids_to_utterance_labels)
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
    embedding_mat1 = np.array([
        [ 1.55329044,  0.82568932,  0.56011276],
        [ 1.10640768, -0.41715366,  0.30323529],
        [ 1.24183824, -2.39021548,  0.02369367],
        [ 1.26094544, -0.27567053,  1.35731148],
        [ 1.59711416, -0.54917262, -0.56074459],
        [-0.4298405 ,  1.39010761, -1.2608597 ]
        ])
    embedding_mat2 = np.array([
        [ 1.63075195,  0.25297823, -1.75406467],
        [-0.59324473,  0.96613426, -0.20922202],
        [ 0.97066059, -1.22315308, -0.37979187],
        [-0.31613254, -0.07262261, -1.04392799],
        [-1.11535652,  0.33905751,  1.85588856],
        [-1.08211738,  0.88559445,  0.2924617 ]
        ])

    # Vector IDs: `vec_ids[i:i+ t]` contains the IDs of embedding[0:t] up to embedding[t - 1:t], with i = t(t - 1)/2
    n_slices = 3 # ??? downsample feature length TODO make equal to length of downsampled feature
    vec_ids = -1*np.ones(int((n_slices**2 + n_slices)/2), dtype=int)
    i_embed = 0
    n_slices_max = 20 # !! also equal to n_landmarks_max in notebook
    for cur_start in range(n_slices):
        for cur_end in range(cur_start, min(n_slices, cur_start + n_slices_max)):
            cur_end += 1
            t = cur_end
            i = int(t*(t - 1)/2)
            vec_ids[i + cur_start] = i_embed
            i_embed += 1
    
    embedding_mats = {}
    vec_ids_dict = {}
    durations_dict = {}
    landmarks_dict = {}
    embedding_mats["test1"] = embedding_mat1
    vec_ids_dict["test1"] = vec_ids
    landmarks_dict["test1"] = [1, 2, 3]
    durations_dict["test1"] = [1, 2, 1, 3, 2, 1]
    embedding_mats["test2"] = embedding_mat2
    vec_ids_dict["test2"] = vec_ids
    landmarks_dict["test2"] = [1, 2, 3]
    durations_dict["test2"] = [1, 2, 1, 3, 2, 1]

    n_rand = np.random.randint(0, 100)
    n_rand = 42
    random.seed(n_rand)
    np.random.seed(n_rand)

    print(f'kmax = {2}, embedding_mats = {embedding_mat1}, vec_ids_dict = {vec_ids}')

    # Initialize model
    landmarks = [1, 2, 3] # frames, utterance starts at frame 0, last frame (3 in this eg) is implicit landmark
    durations = [1, 2, 1, 3, 2, 1] # !! duration of each segment IN FRAMES (10ms units)

    landmarks_ = [0] + landmarks
    N = len(landmarks_)
    durations = -1*np.ones(int(((N - 1)**2 + (N - 1))/2), dtype=int)
    j = 0
    for t in range(1, N):
        for i in range(t):
            if t - i > N - 1:
                j += 1
                continue
            durations[j] = landmarks_[t] - landmarks_[i]
            j += 1
    
    K_max = 3
    segmenter = ESKmeans_Herman( # !! for one utterance at a time instead of stacking all utterances into one matrix
        K_max, embedding_mat1, vec_ids, durations, landmarks, # TODO get landmarks and lengths from relevant scripts
        p_boundary_init=0.5, n_slices_max=2, min_duration=20
        )

    # Perform inference
    def runit():
        segmenter.segment(n_iterations=3)

    print('MY TIME:', timeit.timeit(runit, number=1))

    # obtain clusters and landmarks
    print(segmenter.get_unsup_transcript_i(), segmenter.get_segmented_embeds_i(), segmenter.boundaries)

if __name__ == "__main__":
    main()
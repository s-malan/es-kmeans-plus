"""
Main functions and class for eskmeans segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import numpy as np
from numba import njit
from tqdm import tqdm
from wordseg import downsample
from sklearn.decomposition import PCA

import faiss # https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization

class ESKmeans_Dynamic():
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

    def __init__(self, n_slices_min=0, n_slices_max=20, min_duration=0, p_boundary_init=1.0):

        # Attributes from parameters
        self.n_slices_min = n_slices_min # !! Benji does not use this
        self.n_slices_max = n_slices_max
        self.min_duration = min_duration
        self.p_boundary_init = p_boundary_init
    
    def get_boundaries(self, length):
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
        # Attributes from parameters
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

        self.D = len(self.lengths) # number of utterances

    def get_segmented_embeds_i(self, i):
        """
        Return a list of embedding IDs according to the current segmentation
        for utterance `i`.
        """
        embed_ids = []
        j_prev = 0
        for j in range(self.lengths[i]):
            if self.boundaries[i][j]:
                # We aim to extract seq[j_prev:j+1]. Let the location of this
                # ID be `vec_ids[i, k]`, and we need to find k.
                k = int(0.5*(j + 1)*j)  # this is the index of the seq[0:j] in `vec_ids[i]`
                k += j_prev  # this is the index of the seq[j_prev:j] in `vec_ids[i]`
                embed_ids.append(self.vec_ids[i][k])
                j_prev = j + 1
        return embed_ids
    
    def get_segmented_landmark_indices(self, i=None, boundaries=None, lengths=None):
        """
        Return a list of tuple, where every tuple is the start (inclusive) and
        end (exclusive) landmark index for the segmented embeddings.
        """

        indices = []
        j_prev = 0
        if boundaries is not None and lengths is not None:
            for j in np.where(boundaries[:lengths])[0]:
                indices.append((j_prev, j + 1))
                j_prev = j + 1
        else:
            for j in np.where(self.boundaries[i][:self.lengths[i]])[0]:
                indices.append((j_prev, j + 1))
                j_prev = j + 1
        return indices
    
    def get_vec_embed_neg_len_sqrd_norms(self, embeddings, vec_ids, durations):

        # Get scores
        vec_embed_neg_len_sqrd_norms = -np.inf*np.ones(len(vec_ids))
        indices = -np.inf*np.ones(len(vec_ids))
        for i, embed_id in enumerate(vec_ids):
            if embed_id == -1:
                continue
            
            # find negative L2 distance to closest centroid
            dist, index = self.acoustic_model.index.search(embeddings[embed_id].reshape(1, embeddings.shape[-1]), 1)
            indices[i] = index
            vec_embed_neg_len_sqrd_norms[i] = -1*dist # negative squared L2 distance

            # Scale log marginals by number of frames
            # if np.isnan(durations[i]):
            if durations[i] == -1:
                vec_embed_neg_len_sqrd_norms[i] = -np.inf
                indices[i] = -1
            else:
                vec_embed_neg_len_sqrd_norms[i] *= durations[i]#**self.time_power_term

        return vec_embed_neg_len_sqrd_norms, indices
    
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
    
    def segment_utt_i(self, i):
        """
        Segment new boundaries and cluster new segments for utterance `i`.

        Return
        ------
        sum_neg_len_sqrd_norm : float
            The length-weighted K-means objective for this utterance.
        """

        # sample the current utterance and downsample all possible segmentations
        embeddings = self.data.load_embeddings([self.samples[i]])[0]

        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
            embeddings = embeddings[:, [3,4,5,6,7,8,9,10,12,13,14]]
            
            # Use dense downsample method:
            embeddings = downsample.downsample([embeddings], [self.segments[i]], n=10)

            # Use average as downsampling method:
            # embeddings = np.stack([embeddings[a:b, :].mean(0) for a, b in self.segments[i]])
        else:
            embeddings = downsample.downsample([embeddings], [self.segments[i]], n=10)


        N = self.lengths[i]
        vec_embed_neg_len_sqrd_norms, classes = self.get_vec_embed_neg_len_sqrd_norms(embeddings,
            self.vec_ids[i][:int((N**2 + N)/2)],
            self.durations[i][:int((N**2 + N)/2)]
            )

        # Get new boundaries
        sum_neg_len_sqrd_norm, self.boundaries[i], classes = forward_backward_kmeans_viterbi(
        vec_embed_neg_len_sqrd_norms, classes, N, self.n_slices_min, self.n_slices_max)
        # print('new boundaries after viterbi', self.boundaries[i])

        return sum_neg_len_sqrd_norm, classes

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

        for _ in tqdm(range(n_iterations), desc="Iteration"):

            # Cluster:
            embeddings = []
            # embedding_shapes = []
            for i_utt, (sample, segment) in enumerate(zip(self.samples, self.segments)): # for each utterance sample file path
                if self.pca is not None:
                    embedding = self.pca.transform(self.data.load_embeddings([sample])[0])
                    embedding = embedding[:, [3,4,5,6,7,8,9,10,12,13,14]]
                else:
                    embedding = self.data.load_embeddings([sample])[0]
                segment = list(np.array(segment)[self.get_segmented_embeds_i(i_utt)]) # get only active segments
                
                # Use dense downsample method:
                embeddings.append(downsample.downsample([embedding], [segment], n=10)) # TODO use downsample_utterance if only downsample one utterance
                
                # Use average as downsampling method:
                # embeddings.append(np.stack([embedding[a:b, :].mean(0) for a, b in segment]))
            
            embeddings = np.concatenate(embeddings, axis=0)
            print('Clustering sizes', self.K_max, embeddings.shape)
            self.acoustic_model = faiss.Kmeans(embeddings.shape[1], self.K_max, niter=20, nredo=3, verbose=True)
            self.acoustic_model.train(embeddings)

            # Segment:
            print("Segmenting...")
            classes = []
            sum_neg_len_sqrd_norm = 0
            for i_utt in range(self.D): # get new segments: # TODO do in parallel
                sum_neg_len_sqrd_norm_utt, i_classes = self.segment_utt_i(i_utt)
                sum_neg_len_sqrd_norm += sum_neg_len_sqrd_norm_utt
                classes.append(i_classes)
                
        return sum_neg_len_sqrd_norm, classes

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

@njit
def forward_backward_kmeans_viterbi(vec_embed_neg_len_sqrd_norms, all_classes, N,
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

    boundaries = np.zeros(N, dtype=np.bool_)
    classes = -1*np.ones(N, dtype=np.int_)
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
            print('SHOULD NOT HAPPEN??? WHAT TO DO WITH CLASSES', boundaries)

        q_t = q_t[::-1]
        k = np.argmax(q_t) + 1
        if n_slices_min_cut is not None:
            k += n_slices_min - 1
        
        sum_neg_len_sqrd_norm += vec_embed_neg_len_sqrd_norms[i + t - k]
        # print('here:', i + t - k, vec_embed_neg_len_sqrd_norms[i + t - k])
        classes[t - k] = all_classes[i + t - k] # class of the section starting at t and going back k boundaries
        # print('boundary:', t-k-1, "has class", all_classes[i + t - k])
        if t - k - 1 < 0:
            break
        boundaries[t - k - 1] = True
        t = t - k

    return sum_neg_len_sqrd_norm, boundaries, classes
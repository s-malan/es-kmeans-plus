# Notes on what I need to do:

similar idea to TTI wordseg that the embeddings that are similar will lie close together in the real R-dimensional space. 

each segment is downsampled and flattened to get the embedding, this is so each segment’s embedding is represented by the same number of vectors.

the embeddings are clustered using K-means to find cluster assignments z

Q: look at schematic and if t’s are used as pre-determined hypothesized word boundaries and j’s has a max that it can go back (thats mostly > frame 1). does first segment need to start at j = 1 otherwise it will miss a possible segment and gamma[t] where t = 1, 2 etc (first couple of frames) will not have values. MUSN’T there anyways be a gamma[t] for all t’s since they build on each other? so what about the step-back limit or is it until the previous hypothesized boundary?
A: so t’s and j’s are technically the hypothesized word boundaries. look at schematic on paper

Q: let some wordseg algo do wordseg but let it over-segment quite a lot, try to maximize recall not precision. then use the hypothesized word boundaries as the t’s that can become the actual word boundaries. hopefully all reference boundaries will be hit by one of the hypothesized word boundaries.
A: YES!

Q: what change?
A: other input features instead of MFCCs, use better k-means algo, make more efficient

Q: GradSeg paper gives a bit better results than TTI, should I look into using them as landmarks?
A: run experiment on librispeech, but also they probably evaluate differently so maybe its not actually better…

for initial portential boundaries also look to use vg-HuBERT

get code from:
- Herman repo
  - [Main](https://github.com/kamperh/eskmeans/tree/master)
  - [Feature_Extraction](https://github.com/kamperh/globalphone_awe/blob/master/features/features.py)
  - [Downsample](https://github.com/kamperh/bucktsong_eskmeans/blob/master/downsample/downsample_dense.py)
- Benji DP code
- Hao DP code

if write a paper look at the history of es-kmeans (goes back to 2009 or something, ask Benji)
what we want to talk about: can ESKMeans in the 21st century compete with all these other fancy ASR models or are they all like ESKMeans at the core? can we update ESKMeans to show this (that this idea is the backbone of all ASR models)?
REMEMBER: if you change the number of clusters the model fundamentally changes, so you cannot actually compare across methods with a different number of clusters.
knn_wordseg paper: Aggregation makes a good point saying concatenation preserves the order of features while averaging does not
writing: say eskmeans does best when the landmarks give it a good starting point therefore any landmark algorithm that gives high quality landmarks allows eskmeans to subsample them leading to well defined boundaries

## Facebook k-means

    import faiss
    # Cluster with FAISS
    print(datetime.now())
    print(f"Clustering: K = {args.kmeans}")
    D = X.shape[1]
    kmeans = faiss.Kmeans(
        D, args.kmeans, niter=10, nredo=5, verbose=True, gpu=True
        # D, args.kmeans, niter=1, nredo=1, verbose=True, gpu=True
    )
    kmeans.train(X)

## Repositories

- [Herman](https://github.com/kamperh/eskmeans)
- [Benji]()
  
## Files:
- segment.py : main script to call functions to do the entire eskmeans pipeline
- wordseg
  - landmark_seg.py : functions to get the landmark segments for an utterance using sylseg (later use another wordseg with high recall)
  - embedding.py : functions to extract downsampled acoustic word embeddings for a segment
  - segment.py : main functions and classes for eskmeans segmentation
  - cluster.py : main functions and classes for eskmeans clustering
  - evaluate.py : main functions for evaluation of the segmentation
- utils
  - eskmeans.py : functions to encode and save audio using MFCCs (later HuBERT)
  - data_process.py : functions to sample audio and get corresponding aligments
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

get code from:
- Herman repo
- Benji DP code
- Hao DP code

for initial portential boundaries also look to use vg-HuBERT

if write a paper look at the history of es-kmeans (goes back to 2009 or something, ask Benji)

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
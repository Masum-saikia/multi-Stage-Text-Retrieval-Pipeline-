#Enhancing Q&A Text Retrieval with Ranking Models: Benchmarking, fine-tuning and deploying Rerankers for RAG

This paper discusses how ranking models are used to improve accuracy of the retrieval system, buy benchmarking various publicly available ranking models.
The paper introduces by stating that text retrieval, which is a core component of many retrieval applications, relies heavily on LLMs which are further improved by RAG systems.
Here embedding models are used to convert text into vector representations which captures semantic relationships between words or phrases. They are based on Transformer architecture, and trained using Constrative Learning i.e creating embeddings for pairs of texts and optimizing to make relevant (positive) pairs close in vector space while pushing irrelevant (negative) pairs apart.
Here ranking models are used to order a set of documents, passages, or answers based on their relevance to a given query in text retrieval systems.
Embedding models are implemented as bi-encoders and ranked models are implemented as cross-encoders.
In bi-encoder the query and passage are encoded separately into vectors, and their similarity is measured using methods like cosine similarity or inner product. This allows for efficient large-scale retrieval, but lacks deep interaction between the query and passage.
In cross-encoder the query and passage are concatenated together and jointly encoded by the ranking model, allowing for deeper interactions and better understanding of their semantic relationship. The self-attention mechanism of transformers allows the model to fully capture how tokens in the query relate to tokens in the passage.
The corpus is splited in chunks and transformed into vector representations after than Approximate Nearest Neighbor (ANN) algorithm is used to efficiently find data points (or vectors) that are closest to a given query point in a high-dimensional space.
We also noticed from the MTEB which is a bench mark for text embedding models , that larger the embedding models in parameters more it gets accurate . But larger models requires more computational power, therefore multi-stage text retrieval pipelines is introduced which consist of a embedding model and a ranking model , the embedding model is used to find the top-k candidate passages, after which the ranking models is used to redefine the ranking.
In this paper following models are used for evaluation:
Embedding models
1.Snowflake/snowflake-arctic-embed-l
2.nvidia/nv-embedqa-e5-v5
3.nvidia/nv-embedqa-mistral-7b-v2
Ranking models
1.ms-marco-MiniLM-L-12-v2
2.jina-reranker-v2-base-multilingual
3.mixedbread-ai/mxbai-rerank-large-v1
4.bge-reranker-v2-m3
5.NV-RerankQA-Mistral-4B-v3

11.The general pipline that the paper used is to use datasets from BEIR  datasets, which are already chunked and truncated to max 512 tokens. The chunked passages are embedded using an embedding model and stored in a vector index / database. The querying pipeline then takes place for providing for each query a list with ranked passages for retrieval metrics computation (NDCG@10). In detail, the question is embedded and it is performed a vector search (e.g. using exact or Approximate Nearest Neighbour (ANN) algorithm) on the vector index, returning the top-k most relevant passages for the question. Finally, the top-k (set to 100 in our evaluation experiments) passages are re-ranked with a ranking model to generate the final ordered list.
12. These three Question-Answering datasets from BEIR retrieval benchmark: Natural Questions (NQ) , HotpotQA and FiQA is used for evaluation.
13. NV-RerankQA-Mistral4B-v3 , this model performed best in the benchmark
14. They finetuned the model by keeping only the bottom 16 layers out of its 32 layers, and also modifying its self-attention mechanism from uni-directional (causal) to bi-directional and cange the loss function from binary cross entropy loss to  InfoNCE loss
15. We also learned that larger ranking models provide higher retrieval accuracy,but we can fine tune small models to match the level of accuracy of larger models,the higher retrieval accuracy obtained when using InfoNCE,effectiveness of bi-directional attention for allowing deeper interaction among input query and passage tokens which leads to increase in accuracy.

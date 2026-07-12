"""
Returns the embedding function used to turn text chunks into vectors.

Default: a local Hugging Face sentence-transformers model (BAAI/bge-small-en-v1.5).
This runs 100% locally (downloaded once from Hugging Face, then cached), needs
no API key, and gives strong semantic matching -- so paraphrased questions
and minor typos still find the right chunks instead of only exact keyword
matches.

An AWS Bedrock alternative is left commented below.
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Uncomment to use AWS Bedrock instead:
# from langchain_aws import BedrockEmbeddings


def get_embedding_function():
    # "BAAI/bge-small-en-v1.5" is fast enough for CPU and strong at semantic
    # retrieval. For even better quality (at the cost of a slower first
    # download + slightly slower inference), swap in
    # "BAAI/bge-base-en-v1.5" or "BAAI/bge-large-en-v1.5".
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- AWS Bedrock alternative ---
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )

    return embeddings

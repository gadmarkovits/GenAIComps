# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from langsmith import traceable
from haystack import Document
from fastrag.rankers import IPEXBiEncoderSimilarityRanker, ColBERTRanker

from comps.cores.proto.docarray import SearchedDoc, TextDoc, DocList, BaseDoc
from comps.cores.mega.micro_service import ServiceType, opea_microservices, register_microservice

class RerankedDoc(BaseDoc):
    reranked_docs: DocList[TextDoc]
    initial_query: str

@register_microservice(
    name="opea_service@local_reranking",
    service_type=ServiceType.RERANK,
    endpoint="/v1/reranking",
    host="0.0.0.0",
    port=8000,
    input_datatype=SearchedDoc,
    output_datatype=RerankedDoc,
)
@traceable(run_type="llm")
def reranking(input: SearchedDoc) -> RerankedDoc:
    documents = []
    for i, d in enumerate(input.retrieved_docs):
        documents.append(Document(content=d.text, id=(i + 1)))
    sorted_documents = reranker_model.run(input.initial_query, documents)['documents']
    ranked_documents = [TextDoc(id=doc.id, text=doc.content) for doc in sorted_documents]
    res = RerankedDoc(initial_query=input.initial_query, reranked_docs=ranked_documents)
    return res


if __name__ == "__main__":
    # Use a ColBERT model for ranking 
    # reranker_model = ColBERTRanker("Intel/ColBERT-NQ")

    # To use a BiEncoder model instead, replace the previous line with:
    reranker_model = IPEXBiEncoderSimilarityRanker("Intel/bge-small-en-v1.5-rag-int8-static")
    reranker_model.warm_up()

    opea_microservices["opea_service@local_reranking"].start()

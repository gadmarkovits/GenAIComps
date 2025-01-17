#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -x

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH
    docker build --no-cache -t opea/web-retriever:comps --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/web_retrievers/src/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "opea/web-retriever built fail"
        exit 1
    else
        echo "opea/web-retriever built successful"
    fi
}

function start_service() {

    # tei endpoint
    tei_endpoint=5018
    model="BAAI/bge-base-en-v1.5"
    docker run -d --name="test-comps-web-retriever-tei-endpoint" -e http_proxy=$http_proxy -e https_proxy=$https_proxy -p $tei_endpoint:80 -v ./data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.5 --model-id $model
    export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:${tei_endpoint}"

    # web retriever
    retriever_port=5019
    unset http_proxy
    docker run -d --name="test-comps-web-retriever-server" -p ${retriever_port}:7077 --ipc=host -e GOOGLE_API_KEY=$GOOGLE_API_KEY -e GOOGLE_CSE_ID=$GOOGLE_CSE_ID -e TEI_EMBEDDING_ENDPOINT=$TEI_EMBEDDING_ENDPOINT -e http_proxy=$http_proxy -e https_proxy=$https_proxy opea/web-retriever:comps

    sleep 3m
}

function validate_microservice() {
    retriever_port=5019
    export PATH="${HOME}/miniforge3/bin:$PATH"
    test_embedding=$(python -c "import random; embedding = [random.uniform(-1, 1) for _ in range(768)]; print(embedding)")
    result=$(http_proxy='' curl http://${ip_address}:$retriever_port/v1/web_retrieval \
        -X POST \
        -d "{\"text\":\"What is OPEA?\",\"embedding\":${test_embedding}}" \
        -H 'Content-Type: application/json')
    if [[ $result == *"title"* ]]; then
        echo "Result correct."
    else
        echo "Result wrong. Received status was $result"
        docker logs test-comps-web-retriever-tei-endpoint
        docker logs test-comps-web-retriever-server
        exit 1
    fi
}

function stop_docker() {
    cid_retrievers=$(docker ps -aq --filter "name=test-comps-web*")
    if [[ ! -z "$cid_retrievers" ]]; then
        docker stop $cid_retrievers && docker rm $cid_retrievers && sleep 1s
    fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main

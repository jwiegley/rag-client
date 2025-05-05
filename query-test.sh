#!/bin/bash

unset SSL_CERT_FILE

embedding="HuggingFace:BAAI/bge-large-en-v1.5"
llm="OpenAILike:Falcon3-10B-Instruct"

input=$HOME/org/conference/202410151104-ethdenver-denver-2025.org

# ./rag_client.py                                                 \
#     --embed-model $embedding                                    \
#     --embed-dim 1024                                            \
#     --verbose                                                   \
#     --chunk-size 512                                            \
#     --chunk-overlap 20                                          \
#     --top-k 20                                                  \
#     --from $input                                               \
#     --questions-answered 3                                      \
#     --llm $llm                                                  \
#     --llm-base-url "http://localhost:8080/v1"                   \
#     query                                                       \
#     "What tasks does Hafsah have for the upcoming week?"

case $1 in
    store)
        ./rag_client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-conn "postgresql+psycopg2://postgres@localhost:5432/vector_db" \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            store
        ;;

    llm)
        ./rag_client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-conn "postgresql+psycopg2://postgres@localhost:5432/vector_db" \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            llm
        ;;

    files)
        ./rag_client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-conn "postgresql+psycopg2://postgres@localhost:5432/vector_db" \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            files
        ;;

    query)
        ./rag_client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --db-conn "postgresql+psycopg2://postgres@localhost:5432/vector_db" \
            --streaming                                 \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"                                        \
        ;;

    chat)
        ./rag_client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --timeout 3600                              \
            --max-tokens 1000                           \
            --db-conn "postgresql+psycopg2://postgres@localhost:5432/vector_db" \
            --use-keywords                              \
            --streaming                                 \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"
        ;;

    search)
        ./rag_client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --timeout 3600                              \
            --db-conn "postgresql+psycopg2://postgres@localhost:5432/vector_db" \
            --from $input                               \
            --collect-keywords                          \
            --streaming                                 \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"
        ;;

    *)
        ./rag_client.py                                         \
            --embed-model $embedding                            \
            --embed-dim 1024                                    \
            --verbose                                           \
            --chunk-size 512                                    \
            --chunk-overlap 20                                  \
            --top-k 20                                          \
            --timeout 600                                       \
            --llm $llm                                          \
            --llm-base-url "http://192.168.50.5:8080/v1"        \
            "$@"
        ;;
esac

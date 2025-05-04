#!/bin/bash

unset SSL_CERT_FILE

embedding="HuggingFace:BAAI/bge-large-en-v1.5"
llm="OpenAILike:Falcon3-10B-Instruct"

input=$HOME/org/conference/202410151104-ethdenver-denver-2025.org

# ./rag-client.py                                                 \
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
        ./rag-client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-name "vector_db"                       \
            --db-table "uhj"                            \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            store
        ;;

    llm)
        ./rag-client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-name "vector_db"                       \
            --db-table "uhj"                            \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            llm
        ;;

    files)
        ./rag-client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-name "vector_db"                       \
            --db-table "uhj"                            \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            files
        ;;

    query)
        ./rag-client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --db-name "vector_db"                       \
            --db-table "uhj"                            \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"
        ;;

    chat)
        ./rag-client.py                                 \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --timeout 3600                              \
            --from $input                               \
            --db-name "vector_db"                       \
            --db-table "uhj"                            \
            --llm $llm                                  \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"
        ;;

    *)
        ./rag-client.py                                         \
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

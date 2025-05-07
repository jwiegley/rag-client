#!/bin/bash

unset SSL_CERT_FILE

embedding_provider="HuggingFace"
embedding="BAAI/bge-large-en-v1.5"
llm_provider="OpenAILike"
llm="Falcon3-10B-Instruct"
db_conn="postgresql+psycopg2://postgres@localhost:5432/vector_db"

input=$HOME/org/conference/202410151104-ethdenver-denver-2025.org

case $1 in
    files)
        ./main.py                                       \
            --embed-provider $embedding_provider        \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --from $input                               \
            --questions-answered 3                      \
            --db-conn $db_conn                          \
            --llm-provider $llm_provider                \
            --llm-model $llm                            \
            --llm-base-url "http://localhost:8080/v1"   \
            files
        ;;

    queryold)
            # --source-retries                               \
        ./main.py                                       \
            --embed-provider $embedding_provider        \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --db-conn $db_conn                          \
            --streaming                                 \
            --llm-provider $llm_provider                \
            --llm-model $llm                            \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"                                        \
        ;;

    query)
        ./main.py --config chat.yaml "$@"
        ;;

    chat)
        ./main.py                                       \
            --embed-provider $embedding_provider        \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --timeout 3600                              \
            --max-tokens 1000                           \
            --db-conn $db_conn                          \
            --collect-keywords                          \
            --streaming                                 \
            --llm-provider $llm_provider                \
            --llm-model $llm                            \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"
        ;;

    search)
        ./main.py                                       \
            --embed-provider $embedding_provider        \
            --embed-model $embedding                    \
            --embed-dim 1024                            \
            --verbose                                   \
            --chunk-size 512                            \
            --chunk-overlap 20                          \
            --top-k 20                                  \
            --timeout 3600                              \
            --from $input                               \
            --db-conn $db_conn                          \
            --collect-keywords                          \
            --streaming                                 \
            --llm-provider $llm_provider                \
            --llm-model $llm                            \
            --llm-base-url "http://localhost:8080/v1"   \
            "$@"
        ;;

    *)
        ./main.py                                               \
            --embed-provider $embedding_provider                \
            --embed-model $embedding                            \
            --embed-dim 1024                                    \
            --verbose                                           \
            --chunk-size 512                                    \
            --chunk-overlap 20                                  \
            --top-k 20                                          \
            --timeout 600                                       \
            --llm-provider $llm_provider                        \
            --llm-model $llm                                    \
            --llm-base-url "http://192.168.50.5:8080/v1"        \
            "$@"
        ;;
esac

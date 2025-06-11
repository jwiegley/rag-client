#!/bin/bash

unset SSL_CERT_FILE

db=vector_db
yaml=$PWD/chat.yaml
# input=$HOME/org/conference/202410151104-ethdenver-denver-2025.org
input="$HOME/doc/guidance/local-spiritual-assembly/GLSA/GLSA 2025-02.pdf"

if [[ "$1" == "--reset" ]]; then
    shift 1

    dropdb -U postgres "$db"
    createdb -U postgres -E UTF8 "$db"
    rm -fr ~/.cache/rag-client/*

    ./main.py --from "$input" --config "$yaml" index
fi

./main.py --from "$input" --config "$yaml" "$@"

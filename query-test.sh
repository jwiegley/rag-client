#!/bin/bash

unset SSL_CERT_FILE

export PGHOST=vulcan
export PGPORT=5432
export PGUSER=johnw

db=db
yaml=$PWD/chat.yaml
input="$HOME/doc/guidance/local-spiritual-assembly/GLSA/GLSA 2025-02.pdf"

if [[ "$1" == "--reset" ]]; then
    shift 1
    rm -fr ~/.cache/rag-client/*
    find $HOME/doc/guidance/universal-house-of-justice/messages \
         -name '19630*.txt' -print                              \
         | ./main.py --from - --config "$yaml" index
fi

find $HOME/doc/guidance/universal-house-of-justice/messages \
     -name '19630*.txt' -print                              \
    | ./main.py --from - --config "$yaml" "$@"

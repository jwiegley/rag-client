#!/usr/bin/env bash

KINESIS=$HOME/kadena/kinesis/smart-contracts/pact

find $KINESIS -name '*.pact' -type f -print | while read file; do
    echo "$file" >> ~/Desktop/audit.md
    python3 ./ai.py pact.yaml $file >> ~/Desktop/audit.md
done

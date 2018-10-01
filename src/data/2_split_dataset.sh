#!/bin/bash

# 5 leaves trainset with 80% (i.e., 4/5)
TRAIN_PROPORTION=5
# 2 leaves testset and valset with 50% (i.e., 1/2)
VAL_TEST_PROPORTION=2

if [ "$(pwd)" == "/hear_and_see" ]; then
    # Create valset (20% of trainset)
    cd /hear_and_see/data/torchified/trainset
    echo "Moving files from /hear_and_see/data/torchified/trainset to ../valset"
    find . -type f -exec dirname {} + | uniq -c | while read n d;do echo "Directory:$d Files:$n Moving random:$(($n / $TRAIN_PROPORTION))";mkdir -p ../valset${d:1};find $d -type f | shuf -n $(($n / $TRAIN_PROPORTION)) | while read file;do mv $file ../valset${d:1}/;done;done

    # Create testset (50% of valset)
    cd /hear_and_see/data/torchified/valset
    echo "Moving files from /hear_and_see/data/torchified/valset to ../testset"
    find . -type f -exec dirname {} + | uniq -c | while read n d;do echo "Directory:$d Files:$n Moving random:$(($n / $VAL_TEST_PROPORTION))";mkdir -p ../testset${d:1};find $d -type f | shuf -n $(($n / $VAL_TEST_PROPORTION)) | while read file;do mv $file ../testset${d:1}/;done;done
    cd /hear_and_see/
fi

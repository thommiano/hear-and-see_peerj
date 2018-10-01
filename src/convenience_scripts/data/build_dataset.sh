# Requires UrbanSound8k
# Download here: https://urbansounddataset.weebly.com/download-urbansound8k.html
# Expects hear_and_see/data/UrbanSound8k/

# 1. Restructure UrbanSound8k for Torch's dataloader
# 2. Create validation and testing sets from the trainset created above
. ./src/data/1_torchify_urbansound8k.sh \
    && . ./src/data/2_split_dataset.sh





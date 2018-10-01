# Hear and See: Data Processing

You must first [download UrbanSound8K](https://urbansounddataset.weebly.com/download-urbansound8k.html). Downloading the dataset requires completing a form.

Once you have downloaded the dataset, the remaining scripts expect the data to be available as `hear_and_see/data/UrbanSound8K/`.

## Pipeline

### 1_torchify_urbansound8k.sh

Restructures the UrbanSound8K dataset so that PyTorch's dataloader can access it. Moves files into a new directory: `hear_and_see/data/torchified/trainset`

Run manually by running the following from the project root:

```bash
. ./src/data/1_torchify_UrbanSound8K.sh
```

### 2_split_dataset.sh

Creates `hear_and_see/data/torchified/valset` and `hear_and_/data/torchified/testset` for validation and testing purposes. The validation dataset is used during the training phase and the testing dataset is used during the evaluation phase. The datasets are broken down into the following sample proportions of the original UrbandSound8k dataset:

- 80% training
- 10% validation
- 10% testing

Random sample of the defined proportions are taken from each class folder to retain the original class distributions.

Run manually by running the following from the project root:

```bash
. ./src/data/2_split_dataset.sh
```

### 3_pad_tensors.py

Class observations range from 2-4 seconds in length. Since we convert the sounds into spectrograms, we should make sure they are equal in length so that the spectrograms have equal resolution. Otherwise, CNNs will likely learn to differintiate the classes based on unwanted artifacts.

Padding is accomplished by selecting sounds less than 4 seconds in length and adding frequency at the beginning and end equal to their respective edge frequencies until the clip is 4 seconds.

*Note*: This may be a naive approach that fails to sufficiently thwart the introduction of artifacts.

Run manually by running the following from the project root:

```bash
python ./src/data/3_pad_tensors.py
```

### 4_generate_spectrograms.py

Creates spectrograms from read directory and writes them to write directory.

```bash
python ./src/data/4_generate_spectrograms.py -w 'padded_spectrograms'
```

```bash
python ./src/data/4_generate_spectrograms.py -r 'torchified' -w 'baseline_spectrograms'
```

### 5_roll_spectrograms.py

Padded observations are translated (or "rolled") by -.25, -.50, -.75 and their positive counterparts, which makes the dataset seven times as big. The purpose of this step is to reduce positional bias in some of the observations.

```bash
python ./src/data/5_roll_spectrograms.py
```

Writes to `/hear_and_see/data/rolled_spectrograms/`.
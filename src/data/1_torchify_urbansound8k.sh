DATA_SOURCE_PATH="/hear_and_see/data"

mkdir $DATA_SOURCE_PATH/torchified
mkdir $DATA_SOURCE_PATH/torchified/trainset
mkdir $DATA_SOURCE_PATH/torchified/trainset/air_conditioner $DATA_SOURCE_PATH/torchified/trainset/car_horn $DATA_SOURCE_PATH/torchified/trainset/children_playing $DATA_SOURCE_PATH/torchified/trainset/dog_bark $DATA_SOURCE_PATH/torchified/trainset/drilling $DATA_SOURCE_PATH/torchified/trainset/engine_idling $DATA_SOURCE_PATH/torchified/trainset/gun_shot $DATA_SOURCE_PATH/torchified/trainset/jack_hammer $DATA_SOURCE_PATH/torchified/trainset/siren $DATA_SOURCE_PATH/torchified/trainset/street_music

for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-0-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/air_conditioner/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-1-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/car_horn/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-2-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/children_playing/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-3-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/dog_bark/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-4-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/drilling/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-5-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/engine_idling/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-6-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/gun_shot/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-7-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/jack_hammer/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-8-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/siren/; done
for D in $DATA_SOURCE_PATH/UrbanSound8K/audio/*; do cp $D/*?-9-*?-*?.wav $DATA_SOURCE_PATH/torchified/trainset/street_music/; done

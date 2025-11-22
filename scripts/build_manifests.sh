set -ex
mkdir -p data/{train,test}/{cat,dog}
find PetImages/Cat/ -type f | sort | head -n100 > data/train/cat/manifest.tsv
find PetImages/Dog/ -type f | sort | head -n100 > data/train/dog/manifest.tsv
find PetImages/Cat/ -type f | sort | head -n200 | tail -n100 > data/test/cat/manifest.tsv
find PetImages/Dog/ -type f | sort | head -n200 | tail -n100 > data/test/dog/manifest.tsv
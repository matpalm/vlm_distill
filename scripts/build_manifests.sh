set -ex
mkdir -p data/{train_knn,test_knn}/{cat,dog}
find PetImages/Cat/ -type f | sort | head -n100 > data/train_knn/cat/manifest.tsv
find PetImages/Dog/ -type f | sort | head -n100 > data/train_knn/dog/manifest.tsv
find PetImages/Cat/ -type f | sort | head -n200 | tail -n100 > data/test_knn/cat/manifest.tsv
find PetImages/Dog/ -type f | sort | head -n200 | tail -n100 > data/test_knn/dog/manifest.tsv
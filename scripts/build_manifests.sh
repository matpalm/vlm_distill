set -ex
mkdir -p data/{train_knn,test_knn}/{cat,dog}
find data/Cat/ -type f | sort | head -n100 > data/train_knn/cat/manifest.tsv
find data/Dog/ -type f | sort | head -n100 > data/train_knn/dog/manifest.tsv
find data/Cat/ -type f | sort | head -n200 | tail -n100 > data/test_knn/cat/manifest.tsv
find data/Dog/ -type f | sort | head -n200 | tail -n100 > data/test_knn/dog/manifest.tsv
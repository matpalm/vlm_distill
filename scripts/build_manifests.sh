set -ex

# build manifest files
mkdir -p data/{train_knn,test_knn}
find data/pet_images/cat/ -type f | sort | head -n100 > data/train_knn/manifest.tsv
find data/pet_images/dog/ -type f | sort | head -n100 >> data/train_knn/manifest.tsv
find data/pet_images/cat/ -type f | sort | head -n200 | tail -n100 > data/test_knn/manifest.tsv
find data/pet_images/dog/ -type f | sort | head -n200 | tail -n100 >> data/test_knn/manifest.tsv

# and corresponding label npys
for S in train_knn test_knn; do
 python3 build_labels.py \
  --manifest data/$S/manifest.tsv \
  --labels-npy data/$S/y_true.npy
done
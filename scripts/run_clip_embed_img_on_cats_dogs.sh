set -ex
for S in train test; do
 for L in cat dog; do
  python3 clip_embed_img.py \
   --manifest data/$S/$L/manifest.tsv \
   --npy-output data/$S/$L/clip_embed_img.npy
 done
done

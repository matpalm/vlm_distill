# env setup

for the VLM and clip from hugging face

```
uv venv .hf --python 3.12
source .hf/bin/activate
uv pip install -r requirements.hf.txt
```

for the random embeddings and sklearn

```
uv venv .jax --python 3.12
source .jax/bin/activate
uv pip install -r requirements.jax.txt
```

# baseline zero shot

what is the zero shot performance from `clip(img)` or `clip(text_desc(VLM(img)))` ?

## data

resize everything to 640x640 ( just for simpler modelling later )

```
for L in Cat Dog; do
 find /data/kaggle_cats_and_dogs/PetImages/$L/ -type f > /tmp/manifest
 python3 resize_imgs.py --manifest /tmp/manifest --output-dir data/pet_images/$L/ --hw 640
done
mv data/pet_images/Cat data/pet_images/cat
mv data/pet_images/Dog data/pet_images/dog
```

build manifests for various parts of experiments. #egs => # of imgs for each of cat and dog

split      egs
train_knn  100
test_knn   100

```
./scripts/build_manifests.sh
```

## clip on imgs

CLIP-ViT-B-16; 86M params for img encoder, 63M params for text encoder

* run clip on the imgs from train/test cat/dog to make `clip_embed_img.npy` files
* check knn performance on these zero shot embeddings

```
sh scripts/baseline_clip_img.sh

              precision    recall  f1-score   support
         cat       0.99      1.00      1.00       100
         dog       1.00      0.99      0.99       100
```

## vlm description -> clip on description text

use generic prompt 'describe this image in a sentence`

use `Qwen2.5-VL-7B-Instruct` ( 7B params )

* run vlm on train/test cat/dog to get descriptions ( `p1/descriptions.txt` )
* run clip on these text descriptions to get embeddings ( `p1/clip_embed_text.npy` )
* check knn performance on these zero shot embeddings

```
sh scripts/baseline_vlm_desc_clip_text.sh

              precision    recall  f1-score   support
         cat       0.96      0.99      0.98       100
         dog       0.99      0.96      0.97       100
```

# distilled features from VLM

* same test set as KNN; 100 dog and 100 cat

* v1 baseline training classifier, without teacher; with 5K vs 10K vs 25K imgs

* train classifier, with teacher using
 * embeddings from clip(img) vs clip(vlm_desc(img))
 * on 5K vs 10K vs 25K cats & dogs
 * with (masked) 0, 1, 10, 100K extra images from open images


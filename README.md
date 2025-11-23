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
( includes building a corresponding y_true.npy )

```
split                 #egs
knn/train             100
knn/test              100
cat_dog_1k/train      1000
cat_dog_1k/validate   100
cat_dog_1k/test       100
cat_dog_10k/train     10000
cat_dog_10k/validate  100
cat_dog_10k/test      100
```

```
python build_manifests_and_labels.py
```

## zero shot performance of clip on cats and dogs imgs

CLIP-ViT-B-16; 86M params for img encoder, 63M params for text encoder

* run clip on the imgs from train/test cat/dog to make `clip_embed_img.npy` files
* check knn performance on these zero shot embeddings
* CLIP runs at ~85 imgs / sec

```
sh scripts/baseline_clip_img.sh

              precision    recall  f1-score   support
         cat       0.99      1.00      1.00       100
         dog       1.00      0.99      0.99       100
```

## zero shot performance of clip on VLM descriptions of cats and dogs imgs

use generic prompt 'describe this image in a sentence`

use `Qwen2.5-VL-7B-Instruct` ( 7B params )

* run vlm on train/test cat/dog to get descriptions ( `p1/descriptions.txt` )
* run clip on these text descriptions to get embeddings ( `p1/clip_embed_text.npy` )
* check knn performance on these zero shot embeddings
* VLM runs at about ~1.5imgs / sec

```
sh scripts/baseline_vlm_desc_clip_text.sh

              precision    recall  f1-score   support
         cat       0.96      0.99      0.98       100
         dog       0.99      0.96      0.97       100
```

# distilled features from VLM

## how small a model can we train to replicate the zero shot clip embeddings?

* train a model to replicate embeddings from clip on `cat_dog_1K` images
* use that model to embed `train_knn`
* train KNN and check performance on `test_knn`

* train classifier, with teacher using
 * embeddings from clip(img) vs clip(vlm_desc(img)) ( generic prompt ) vs clip(vlm_desc(img)) ( specific prompt )
 * on increasing sizes of cat vs dog
 * with (masked) 0, 1, 10, 100K extra images from open images

where
 generic prompt => "describe this image in a sentence."
 specific prompt => "describe the features of this image in the context of deciding if it is a cat, or a dog, or something else."

TODOS

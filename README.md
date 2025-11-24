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
./scripts/resize_cats_and_dogs.sh
```

build manifests for various parts of experiments. #egs => # of imgs for each of cat and dog
( includes building a corresponding y_true.npy )

```
python build_manifests_and_labels.py
```

results in distinct sets from cats and dogs

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

## zero shot performance of clip on cats and dogs imgs

CLIP-ViT-B-16; 86M params for img encoder, 63M params for text encoder

* run clip on the imgs from train/test cat/dog to make `clip_embed_img.npy` files
* check knn performance on these zero shot embeddings
* CLIP runs at ~85 imgs / sec

```
./scripts/baseline_clip_img.sh

              precision    recall  f1-score   support
         cat       1.00      1.00      1.00       100
         dog       1.00      1.00      1.00       100
```

## zero shot performance of clip on VLM descriptions of cats and dogs imgs

use generic prompt 'describe this image in a sentence`

use `Qwen2.5-VL-7B-Instruct` ( 7B params )

* run vlm on train/test cat/dog to get descriptions ( `p1/descriptions.txt` )
* run clip on these text descriptions to get embeddings ( `p1/clip_embed_text.npy` )
* check knn performance on these zero shot embeddings
* VLM runs at about ~1.5imgs / sec

```
./scripts/baseline_vlm_desc_clip_text.sh

              precision    recall  f1-score   support
         cat       1.00      1.00      1.00       100
         dog       1.00      1.00      1.00       100
```

# distilled features from VLM

## how small a model can we train to replicate the zero shot clip embeddings?

we'll train a model on 1K ( or 10K ) images to replicate the embeddings from clip and the VLM

start by embedding the cat_dog_1k and cat_dog_10k datasets imgs with clip

```
./scripts/embed_other_imgs_clip.sh
```

then describe the cat_dog_1k via the VLM with two prompts, and embed those descriptions with clip

first prompt is generic; "describe this image in a sentence"

second prompt is more specific to this task; "describe the primary features of this image, in a single sentence,
with respect to classifying the image as a cat, or a dog, or neither."

```
./scripts/embed_other_imgs_vlm_desc_clip_text.sh
```

this gives us 4 training sets of (img, embeddings)

imgs          embeddings from
cat_dog_1k    clip_img
cat_dog_1k    clip_text(vlm_p1)
cat_dog_1k    clip_text(vlm_p2)
cat_dog_10k   clip_img

we can train models to replicate these embeddings.
after each epoch use the trained model to generate embeddings for the zero shot KNN task tested aboveclip

## including open images data

sample 100K of open images and resize to 640x640 ( non squash )

```
./scripts/sample_and_resize_open_images.sh
```
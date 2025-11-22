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

# manifest

400 images; 100 each of

* train, cat
* train, dog
* validate, cat
* validate, dog

```
sh scripts/build_manifests.sh
```

## baseline zero shot

what is the zero shot performance from `clip(img)` or `clip(text_desc(VLM(img)))` ?

### clip on imgs

CLIP-ViT-B-16; 86M params for img encoder, 63M params for text encoder

* run clip on the imgs from train/test cat/dog to make `clip_embed_img.npy` files
* check knn performance on these zero shot embeddings

```
sh scripts/baseline_clip_img.sh

              precision    recall  f1-score   support
         cat       0.99      1.00      1.00       100
         dog       1.00      0.99      0.99       100
```

### vlm description -> clip on description text

use generic prompt 'describe this image in a sentence`

* run vlm on train/test cat/dog to get descriptions ( `p1/descriptions.txt` )
* run clip on these text descriptions to get embeddings ( `p1/clip_embed_text.npy` )
* check knn performance on these zero shot embeddings

```
sh scripts/baseline_vlm_desc_clip_text.sh

              precision    recall  f1-score   support
         cat       0.96      0.99      0.98       100
         dog       0.99      0.96      0.97       100
```

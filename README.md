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

3s for the clip embeddings of the images

```
# scripts/run_clip_embed_img_on_cats_dogs.sh
for S in train test; do
 for L in cat dog; do
  python3 clip_embed_img.py \
   --manifest data/$S/$L/manifest.tsv \
   --npy-output data/$S/$L/clip_embed_img.npy
 done
done
```

```

### vlm description -> clip on description text

~4m for the vlm descriptions; + 3s for the clip embeddings

```
for S in train test; do
 for L in cat dog; do
  python3 vlm_describe.py \
   --manifest data/$S/$L/manifest.tsv \
   --prompt 'describe this image in a sentence' \
   --txt-output data/$S/$L/vlm_describe_prompt_1.txt
  python3 clip_embed_text.py \
   --text data/$S/$L/vlm_describe_prompt_1.txt \
   --npy-output data/$S/$L/clip_embed_vlm_desc_1.npy
 done
done
```
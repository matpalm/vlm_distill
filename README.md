./build_manifest.sh

# embeddings v1; baseline random projections
# TODO

# embeddings v2; clip on imgs
python3 clip_embed_img.py \
 --manifest manifest.tsv \
 --npy-output clip_embed_img.npy

# embeddings v3; vlm description -> clip on txt
python3 vlm_describe.py \
 --manifest manifest.tsv \
 --prompt 'describe this image in a sentence' \
 --txt-output vlm_describe_prompt_1.txt
python3 clip_embed_text.py \
 --text vlm_describe_prompt_1.txt \
 --npy-output clip_embed_vlm_desc_1.npy

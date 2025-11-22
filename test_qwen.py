from models import VLM

vlm = VLM()
result = vlm.prompt(
    prompt="Describe the following image using a short paragraph.",
    img_path="imgs/Selection_073.png",
)
print(result)

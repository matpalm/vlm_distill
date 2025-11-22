from models import VLM

vlm = VLM()
description = vlm.describe("imgs/Selection_073.png")
print(description)

# OBJECTS IN A COMPLEX IMAGE — detect and list them.
#
# document_classifier.py used CLIP for ONE label on a whole image. That falls
# apart on a busy scene: CLIP scores one vector for the entire photo, so small
# or repeated subjects never stand out.
#
# OWL-ViT is open-vocabulary DETECTION instead: give it text queries, get back
# one box + score per instance it actually finds. We just list the hits above a
# score threshold — no counting (rose bushes explode into "flowers"), no boxes
# drawn, just "what's in here and how sure".
#
# run:  cd llamaindex && python objects_in_image.py
#       first run downloads google/owlvit-base-patch32 (~600 MB); CPU, ~1 min

import _trace  # noqa: F401  -- loads practice/.env (folder convention; nothing to trace here)

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

IMAGE = "images/complex.jpg"
SCORE_THRESHOLD = 0.25       # OWL-ViT scores run low; raise to be stricter

VOCAB = [
    "a man", "a woman", "a dog", "a cat", "a bird", "a squirrel", "a butterfly",
    "a palm tree", "a rose bush", "a shrub", "a potted plant",
    "a stone wall", "a wooden fence", "a wicker basket", "a straw hat",
    "a garden bed", "a gravel path",
]

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval()

image = Image.open(IMAGE).convert("RGB")
inputs = processor(text=[VOCAB], images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

result = processor.post_process_grounded_object_detection(
    outputs,
    threshold=SCORE_THRESHOLD,
    target_sizes=torch.tensor([image.size[::-1]]),   # (height, width)
    text_labels=[VOCAB],
)[0]

hits = sorted(
    zip((s.item() for s in result["scores"]), result["text_labels"]),
    reverse=True,
)

print(f"{IMAGE} - {len(hits)} objects detected (score >= {SCORE_THRESHOLD}):\n")
for score, label in hits:
    print(f"  {score:.2f}  {label}")

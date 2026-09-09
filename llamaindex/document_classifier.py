# DOCUMENT CLASSIFIER — label every image in images/ with CLIP zero-shot.
#
# CLIP embeds an image and a few short text prompts into the SAME vector space.
# Score the image against one prompt per class, sharpen with a temperature, then
# softmax -> a probability per class. Highest one wins. No training, no labels.
#
# run:  cd llamaindex && python document_classifier.py
#       first run downloads clip-ViT-B-32 (~350 MB)

import _trace  # noqa: F401  -- loads practice/.env (folder convention; nothing to trace here)

from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer, util

# class -> the prompt CLIP compares each image against
CLASSES = {
    "bird": "a photo of a bird",
    "mammal": "a photo of a mammal",
    "human": "a photo of a person",
}

model = SentenceTransformer("clip-ViT-B-32")

image_paths = sorted(Path("images").glob("*.jpg"))

# encode all the images and all the class prompts into the same vector space
image_vecs = model.encode([Image.open(p) for p in image_paths], normalize_embeddings=True)
# encode the class prompts into vectors, normalised to unit length. CLIP's
label_vecs = model.encode(list(CLASSES.values()), normalize_embeddings=True)


labels = list(CLASSES)

# compute the cosine similarity between every image and every class prompt
sims = util.cos_sim(image_vecs, label_vecs)     # (n_images, n_classes), cosine
probs = (sims * 100).softmax(dim=1)             # *100 = CLIP's temperature, then normalise

print(f"{'file':9}{'label':9}{'conf':6}per-class probabilities")
for i, path in enumerate(image_paths):
    top = int(probs[i].argmax())
    spread = "  ".join(f"{labels[j]}={probs[i][j]:.2f}" for j in range(len(labels)))
    print(f"{path.name:9}{labels[top]:9}{probs[i][top]:.2f}  {spread}")

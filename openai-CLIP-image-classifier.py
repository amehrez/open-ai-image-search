import torch

import clip

from PIL import Image

import os


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


image_folder = "C:/Users/Ahmad/Desktop/"

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(
    ('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]


text = clip.tokenize(["dancers", "singers", "protesters"]).to(device)


for image_file in image_files:

    image_path = os.path.join(image_folder, image_file)

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():

        image_features = model.encode_image(image)

        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print(f"Label probs for {image_file}:", probs)

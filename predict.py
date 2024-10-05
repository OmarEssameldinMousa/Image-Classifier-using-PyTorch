import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

# Command line arguments
parser = argparse.ArgumentParser(description="Predict flower name from an image")
parser.add_argument('image_path', type=str, help='Path to the image')
parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--top_k', type=int, default=1, help='Top K most likely classes')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping from category label to name')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

model = load_checkpoint(args.checkpoint)

# Process Image
def process_image(image_path):
    # I have tried different implementation other than the one that exist in the
    # notebook to be familiar with different ways to process the image
    img = Image.open(image_path)
    img = img.resize((256, 256)).crop((16, 16, 240, 240))
    
    np_img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    np_img = np_img.transpose((2, 0, 1))
    
    return torch.tensor(np_img).float().unsqueeze(0)

# Prediction function
def predict(image_path, model, topk=5):
    model.eval()
    img_tensor = process_image(image_path)
    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logps = model(img_tensor)
    
    ps = torch.exp(logps)
    top_p, top_classes = ps.topk(topk, dim=1)
    
    top_p = top_p.cpu().numpy()[0]
    top_classes = top_classes.cpu().numpy()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_classes]
    
    return top_p, top_classes

# Load category names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Make prediction
probs, classes = predict(args.image_path, model, args.top_k)
flower_names = [cat_to_name[str(cls)] for cls in classes]

# Print results
for prob, name in zip(probs, flower_names):
    print(f"{name}: {prob:.4f}")

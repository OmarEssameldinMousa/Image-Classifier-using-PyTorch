import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Command line arguments
parser = argparse.ArgumentParser(description="Train a model for flower classification")
parser.add_argument('--data_dir', type=str, default='flowers', help='Dataset directory')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--hidden_units', type=int, default=2048, help='Number of hidden units in the classifier')
parser.add_argument('--save_dir', type=str, default='checkpts/checkpoint.pth', help='Save directory for the model checkpoint')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18'], help='Model architecture')

args = parser.parse_args()
print("Script started..")

# Directories for train, valid datasets
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

# Required Transforms for the datasets
print("Transforming datasets..")
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Loading datasets from directories
print("Loading datasets from directories..")
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

# Defining dataloaders for train and valid datasets
print("Defining dataloaders for train and valid datasets..")
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False)
}

# Loading the pre trained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif args.arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    input_size = model.fc.in_features  # ResNet18 output size is 512

# Freezing parameters
for param in model.parameters():
    param.requires_grad = False

# Defining the classifier
classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(args.hidden_units, 102),  # Assuming 102 flower classes
    nn.LogSoftmax(dim=1)
)

if args.arch == 'vgg16':
    model.classifier = classifier
elif args.arch == 'resnet18':
    model.fc = classifier

# Loss function and optimizer
criterion = nn.NLLLoss()
if args.arch == 'vgg16':
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
elif args.arch == 'resnet18':
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

# Device setup: GPU or CPU
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Training 
print("\n", "*"*10, " Training start ", "*"*10)
print_every = 20
for epoch in range(args.epochs):
    running_loss = 0
    step = 0
    model.train()  
    for inputs, labels in dataloaders['train']:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  
        logps = model(inputs)  # Forward pass
        loss = criterion(logps, labels)  # Loss calculation
        loss.backward()  # Backpropagation
        optimizer.step()  

        running_loss += loss.item()
        if step % print_every == 0 or step == 1 or step == len(dataloaders['train']):
            print(f"Epoch: {epoch+1}/{args.epochs} Batch % Complete {(step)*100/len(dataloaders['train']):.2f}%")

    # Validation 
    model.eval()  
    validation_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)
            batch_loss = criterion(logps, labels)
            validation_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train Loss: {running_loss/len(dataloaders['train']):.3f}.. "
          f"Validation Loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
          f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")
      
print("\n", "*"*10, " Training Ended ", "*"*10)

# Saving the model
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'classifier': model.classifier if args.arch == 'vgg16' else model.fc,
    'arch': args.arch  
}
try:
    torch.save(checkpoint, args.save_dir)
    print(f"Saving model to {args.save_dir}")
except Exception as e:
    print(f"Error saving the model: {e}")

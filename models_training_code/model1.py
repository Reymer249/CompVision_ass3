#!/usr/bin/env python
# coding: utf-8

# # Notes
# 
# 1) Add preprocessing transformation - DONE!

# # Imports & Constants

# In[1]:


from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import v2
import numpy as np
from tqdm import tqdm
import pickle
from torchsummary import summary


# In[2]:


SEED = 42
NUM_WORKERS = 16
BATCH_SIZE = 8
NUM_SEGMENTS = 8
RES_NEXT_OUT = 2048
NUM_EPOCHS = 20
CHECKPOINT_FOLDER = os.path.join('models_checkpoints', 'model_2_tochno')
METRICS_FOLDER = os.path.join('metrics', 'model_2_tochno')

np.random.seed(SEED)


# In[3]:


LABELS_PATH = "jester-v1-labels.csv"
TRAIN_LABELS = "train.csv"
VAL_LABLES = "val.csv"
TEST_LABELS = "test.csv"
DATA_ROOT = "20bn-jester-v1"


# Load the pretrained ResNeXt101_32x8d model to use it for feature extraction \
# We load it here as we get the transformation function from it. 

# In[4]:


weights = models.ResNeXt101_32X8D_Weights.DEFAULT
res_next = models.resnext101_32x8d(weights=weights)
res_next.eval()
res_next = nn.Sequential(*list(res_next.children())[:-1])
# Freeze all layers so that they are not updated during training
for param in res_next.parameters():
    param.requires_grad = False
    
# Get the transformations needed for the model
preprocess_transform = weights.transforms()


# # Define the datasets and data loaders

# ## Functions

# In[5]:


with open(LABELS_PATH) as labels_file:
    labels = labels_file.readlines()
    #labels = [label[:-1] for label in labels]
    labels_encode_dict = dict(zip(labels, range(len(labels))))
    labels_decode_dict = dict(zip(range(len(labels)), labels))


# In[6]:


class VideoDataset(Dataset):
    def __init__(self, root_dir, split, label_dict, transform=None, n_segments=None, frame_limit=None):
        """
        Initialize the dataset with the root directory for the videos,
        the split (train/val/test), an optional data transformation,
        and an optional label dictionary.

        Args:
            root_dir (str): Root directory for videos
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_dict = label_dict
        self.frame_limit = frame_limit
        self.n_segments = n_segments
        self.videos_paths = []
        self.labels_num = []
        self.labels_str = []

        with open(self.split + '.csv') as r:
            lines = r.readlines()
            for line in lines:
                line = line.split(';')
                self.videos_paths.append(line[0])
                self.labels_num.append(label_dict[line[1]])
                self.labels_str.append(line[1])


    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.labels_num)

    
    @staticmethod
    def _select_frames(list_of_frames, num_segments):
        n = len(list_of_frames)
        segment_boundaries = np.linspace(0, n, num_segments+1, dtype=int)  # Define segment boundaries
        selected_indices = [np.random.randint(segment_boundaries[i], segment_boundaries[i + 1]) 
                            for i in range(num_segments)]  # Sample 1 index per segment
        selected_frames = [list_of_frames[i] for i in selected_indices]  # Map indices to frames
    
        return selected_frames
        

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.videos_paths[idx])
        label = self.labels_num[idx]
    
        # Load all frames in the video
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])
        if self.frame_limit:
            frame_files = frame_files[:self.frame_limit]

        if self.n_segments:
            frame_files = self._select_frames(frame_files, self.n_segments)
    
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame = Image.open(frame_path).convert("RGB")
            if self.transform:
                frame = self.transform(frame)  # Apply transform to convert to tensor
            else:
                frame = transforms.ToTensor()(frame)  # Default conversion if no transform provided
            frames.append(frame)
    
        # Stack frames into a tensor (T x C x H x W)
        video_tensor = torch.stack(frames)
    
        return video_tensor, label


# ## Define datasets and loaders

# In[7]:


transformations_train = v2.Compose([
    v2.RandomApply([v2.ElasticTransform(alpha=50.0, sigma=9.0)], p=0.2),
    v2.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.1
    ),
    v2.RandomAdjustSharpness(sharpness_factor=2),
    v2.RandomAutocontrast(),
    v2.RandomEqualize(),
    preprocess_transform
])


# In[8]:


train_dataset = VideoDataset(DATA_ROOT, "train", labels_encode_dict, transform=transformations_train, n_segments=NUM_SEGMENTS)
val_dataset = VideoDataset(DATA_ROOT, "val", labels_encode_dict, transform=preprocess_transform, n_segments=NUM_SEGMENTS)
test_dataset = VideoDataset(DATA_ROOT, "test", labels_encode_dict, transform=preprocess_transform, n_segments=NUM_SEGMENTS)


# In[9]:


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)


# In[10]:


subset_indices = torch.randperm(len(train_dataset)).tolist()[:100]  # Select 100 samples randomly
train_small = Subset(train_dataset, subset_indices)
subset_indices = torch.randperm(len(val_dataset)).tolist()[:100]  # Select 100 samples randomly
val_small = Subset(val_dataset, subset_indices)


# In[11]:


train_small_loader = DataLoader(
    train_small,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
)

val_small_loader = DataLoader(
    val_small,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
)


# # Model architecture

# ResNeXt output - 2048 features

# In[12]:


class GestureClassifier(nn.Module):
    def __init__(self, input_size, num_classes):

        super().__init__()

        # define 
        self.ln1 = nn.Linear(input_size, int(input_size/4))
        self.ln2 = nn.Linear(int(input_size/4), int(input_size/8))
        self.ln3 = nn.Linear(int(input_size/8), num_classes)

        # init
        self.initialize_layer(self.ln1)
        self.initialize_layer(self.ln2)
        self.initialize_layer(self.ln3)


    def forward(self, x):
        x = torch.relu(self.ln1(x))
        x = torch.relu(self.ln2(x))
        x = self.ln3(x)
        return x
        

    @staticmethod
    def initialize_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


# In[13]:


model = GestureClassifier(input_size = RES_NEXT_OUT * NUM_SEGMENTS, num_classes=len(labels_encode_dict))
device = "cuda"
model.to(device)
summary(model, input_size = (RES_NEXT_OUT*NUM_SEGMENTS,))


# # Training function

# In[14]:


def process_video_batch(res_next, video_batch):
    """
    Process video frames and extract features using res_next.
    Args:
        res_next: Pretrained feature extractor (e.g., ResNeXt).
        video_batch: Tensor of shape [batch_size, num_frames, 3, 224, 224].

    Returns:
        Concatenated features for each video: [batch_size, num_frames * feature_dim].
    """
    batch_size, num_frames, c, h, w = video_batch.shape

    # Reshape to process frames independently
    frames = video_batch.view(batch_size * num_frames, c, h, w)  # [batch_size * num_frames, 3, 224, 224]
    
    # Extract features for each frame
    frame_features = res_next(frames)  # Output shape: [batch_size * num_frames, feature_dim]
    
    # Reshape back to group frames for each video
    frame_features = frame_features.view(batch_size, num_frames, -1)  # [batch_size, num_frames, feature_dim]
    
    # Concatenate features along the temporal dimension
    fused_features = frame_features.view(batch_size, -1)  # [batch_size, num_frames * feature_dim]
    
    return fused_features


def evaluate(model, features_model, eval_loader, criterion, device):
    """
    Evaluate the classifier on the validation set.

    Args:
        model (CNN): CNN classifier to evaluate.
        features_model: CNN to extract features from images. 
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in eval_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits and loss
            logits = model(process_video_batch(features_model, inputs))
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)


    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy

def train(model, features_model, train_loader, val_loader, optimizer, criterion, device,
          num_epochs, with_train_set_metrics=False):
    """
    Train the classifer on the training set and evaluate it on the validation set every epoch.

    Args:
    model (CNN): classifier to train.
    features_model: CNN to extract features from images. 
    train_loader (torch.utils.data.DataLoader): Data loader for the training set.
    val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    criterion (callable): Loss function to use for training.
    device (torch.device): Device to use for training.
    num_epochs (int): Number of epochs to train the model.
    """

    # Place the model on device
    model = model.to(device)
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        model.train() # Set model to training mode

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch +1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                #Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Compute the logits and loss
                logits = model(process_video_batch(features_model, inputs))
                loss = criterion(logits, labels)

                # Update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
            avg_loss, accuracy = evaluate(model, features_model, val_loader, criterion, device)
            print(
                f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
                )
            if with_train_set_metrics:
                train_avg_loss, train_accuracy = evaluate(model, features_model, train_loader, criterion, device)
                print (
                    f'Train set: Average loss = {train_avg_loss:.4f}, Accuracy = {train_accuracy:.4f}'
                )
                losses.append((train_avg_loss, avg_loss))
                accuracies.append((train_accuracy, accuracy))
            else:
                losses.append(avg_loss)
                accuracies.append(accuracy)
            with open(os.path.join(METRICS_FOLDER, 'losses.pkl'), 'wb') as f:
                pickle.dump(losses, f)
            with open(os.path.join(METRICS_FOLDER, 'accuracies.pkl'), 'wb') as f:
                pickle.dump(accuracies, f)
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict()
                },
                os.path.join(CHECKPOINT_FOLDER, f'model_{epoch+1}_out_of_{num_epochs}.ckpt')
            )

        # plt.clf()  # Clear the current figure
        # plt.plot(losses[:, 0], label='Training Loss')
        # plt.plot(losses[:, 1], label='Validation Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()
        # plt.pause(0.1)  # Pause to update the plot
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            }, 
            os.path.join(CHECKPOINT_FOLDER, 'model.ckpt')
        )

    return losses, accuracies

def test(model, features_model, test_loader, device):
    """
    Get predictions for the test set.

    Args:
        model (CNN): classifier to evaluate.
        features_model: CNN to extract features from images. 
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model = model.to(device)
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        all_preds = []

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)

            logits = model(process_video_batch(features_model, inputs))

            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)
    return all_preds

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy


# # Training

# Define the parameters of training

# In[15]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureClassifier(input_size = RES_NEXT_OUT * NUM_SEGMENTS, num_classes=len(labels_encode_dict))
model.to(device)
res_next.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# In[17]:


train(model, res_next, train_loader, val_loader, optimizer, criterion, device, NUM_EPOCHS)


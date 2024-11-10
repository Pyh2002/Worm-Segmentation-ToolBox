import segmentation_models_pytorch as smp
import os
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


class SegmentationDataset(Dataset):

    def __init__(self, image_folder, mask_folder, transform=None):
        """ Segmentation dataset constructor.

        Initializes a dataset consisting of .png images of both the original image and its corresponding mask.
        Both the original image and the mask must have the same filename.

        Args:
            image_folder (string): absolute filepath to the folder of Images
            mask_folder (string): absolute filepath to the folder of Masks
            transform (Transform): a Transform object consisting of all the transforms


        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        #Can swap to image types other than .png if needed
        self.image_filenames = [filename for filename in os.listdir(image_folder) if filename.endswith(".png")]
        self.mask_filenames = [filename for filename in os.listdir(mask_folder) if filename.endswith(".png")]

    def __len__(self):
        """ Returns the number of images in the dataset

        Returns:
            int: number of images in the dataset

        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """ Opens the image and mask at the given index. The image and mask is opened as a greyscale PIL image

        Args:
            idx: (int) index of the image in the dataset to be acquired

        Returns:
            (Image,Image): a tuple with the original image as the first value and the mask as the second

        """
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_folder, self.mask_filenames[idx])

        image = Image.open(img_name).convert("L")
        mask = Image.open(mask_name).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
def train_epoch(model, dataloader,loss_fn, optimizer, device, scaler):
    """ Trains the model going through the training data one time (epoch)

    Args:
        model: (nn.Module) The pyTorch neural network model
        dataloader: (DataLoader) a DataLoader object corresponding to the training Dataset
        loss_fn: (Callable) a function to quantify how well the model fits the dataset
        optimizer: (Optimizer) an optimizer object in charge of refining the model to best fit the dataset
        device: (device) the device on which training will be performed. Should probably be an Nvidia GPU
        scaler: (scaler) gradient scaler used for Nvidia Automatic Mixed Precision

    Returns:
        float: the loss value from the current epoch

    """
    model.train()  # Set model to training mode
    running_loss = 0.0  # Initialize loss
    progress = tqdm(dataloader, desc="Training")  # Initialize progress bar

    for batch, (images, masks) in enumerate(dataloader):  # Loop through training data
        optimizer.zero_grad()  # Reset gradients
        images = torch.stack(images).to(device)  # Send images to the device
        masks = torch.stack(masks).to(device)  # Send masks to the device
        # Nvidia Automatic Mixed Precision code
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(images)
            assert outputs.dtype is torch.float16
            loss = loss_fn(outputs, masks)
            assert loss.dtype is torch.float32
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()  # Sum loss values from each image
        progress_bar_dict = dict(loss=loss.item(), avg_loss=running_loss / (batch+1))  # Add values to progress bar
        progress.set_postfix(progress_bar_dict)  # Put new values on progress bar
        progress.update()  # Update progress bar

    progress.close()  # Close progress bar
    epoch_loss = running_loss / (batch+1)  # Calculate average loss

    return epoch_loss  # Return average loss


def evaluate_model(model, dataloader,loss_fn, device, metric):
    """ Evaluates the model against the validation dataset once (epoch)

    Args:
        model: (nn.Module) The pyTorch neural network model
        dataloader: (DataLoader) a DataLoader object corresponding to the validation Dataset
        loss_fn: (Callable) a function to quantify how well the model fits the dataset
        device: (device) the device on which validation will be performed. Should probably be a Nvidia GPU
        metric: (Callable) another metric for evaluating model performance

    Returns:
        float: the loss value from the current epoch
    """
    model.eval()  # Set model to validation mode
    running_loss = 0.0  # Initialize loss value
    running_metric = 0.0  # Initialize metric value
    progress = tqdm(dataloader, desc="Evaluate")  # Initialize progress bar
    with torch.no_grad():  # Prevent gradients from being modified
        for batch, (images, masks) in enumerate(dataloader):
            images = torch.stack(images).to(device)  # Send images to the device
            masks = torch.stack(masks).to(device)    # Send masks to the device

            # Nvidia Automatic Mixed Precision code
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)

            loss = loss_fn(outputs, masks)  # Calculate loss for the current image
            intersection = metric(outputs, masks)  # Calculate the metric for the current image
            running_loss += loss.item()  # Sum loss for average
            running_metric += intersection.item()  # sum metric for average
            # Add values to progress bar
            progress_bar_dict = dict(loss=loss.item(), avg_loss=running_loss / (batch+1), IoU = running_metric/(batch+1))
            progress.set_postfix(progress_bar_dict)  # Put new values on progress bar
            progress.update()  # Update progress bar
    epoch_loss = (running_loss / (batch+1),running_metric/(batch+1))  # Create tuple for loss and the metric
    return epoch_loss  # Return the loss

def collate_fn(batch):
    """ A function for ordering the batch as a tuple
    Args:
        batch: (batch) the current batch being run.

    Returns:
        tuple: tuple of batch items

    """
    return tuple(zip(*batch))


if __name__ == '__main__':
    #Actually training and validating the model needs to be under main so that it does not run twice

    # Device chosen for training and validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    val_folder_masks = "Training/Val/Masks"  # Filepath to validation masks
    val_folder_images = "Training/Val/Images"  # Filepath to validation images
    train_folder_masks = "Training/Train/Masks"  # Filepath to training masks
    train_folder_images = "Training/Train/Images"  # Filepath to training images
    # Image/Mask transforms before being fed into the model
    data_transform = transforms.Compose([  # Combine multiple transformations
        # transforms.Resize((1024, 1024)),  # Resize to desired dimensions 
        transforms.ToTensor(),  # Convert to a tensor
    ])
    train = SegmentationDataset(train_folder_images, train_folder_masks,data_transform)  # Create training dataset object
    valid = SegmentationDataset(val_folder_images, val_folder_masks,data_transform)  # Create validation dataset object
    data_loader_params = {
        'batch_size': 10,  # Number of images in the batch. Seems to affect results. Increasing can cause memory issues
        'num_workers': 16,  # Number of CPU cores allocated to the multiprocessing of the batch
        'persistent_workers': True,  # Keeps dataset instances alive
        'pin_memory': True,  # Speeds up data transfer to CUDA GPUs
        'pin_memory_device': 'cuda',  # Device to pin memory to. Should be the GPU.
        'collate_fn': collate_fn,  # Function to organize images into mini batches of tensors
    }
    # Create data loaders
    training = DataLoader(train, **data_loader_params, shuffle=True)  # Shuffling the data helps with training
    validation = DataLoader(valid, **data_loader_params, shuffle=False)  # Don't shuffle on the validation set

    #  Create a neural network model
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    )

    model = model.to(torch.device('cuda'))  # Send the model to the GPU
    # Define loss function and optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)  # lr is the learning rate, too high causes divergence

    loss_fn = nn.BCEWithLogitsLoss().cuda()  # Create the loss function and send it to the GPU
    jaccard = BinaryJaccardIndex().cuda()  # Create the metric function and send it to the GPU

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None  # Create the gradient scaler for Nvidia AMP
    best_loss = float('inf')  # Initialize the best validation loss
    best_metric = 0.0  # Initialize the best metric

    epochs = 30  # Number of epochs to train/validate

    progress = tqdm(range(epochs), desc="Epochs")  # Progress bar for Epochs

    for epoch in progress:  # loop through the epochs
        train_loss = train_epoch(model, training,loss_fn, optimizer, device, scaler)  # Train the model through an epoch
        val_loss = evaluate_model(model, validation,loss_fn, device, jaccard)  # Validate the model through an epoch

        if val_loss[0] < best_loss:  # Identify the best validation loss
            best_loss = val_loss[0]  # Update best validation loss
            torch.save(model.state_dict(), 'best.pth')  # Save the state dictionary of the model for the best loss value
        if val_loss[1] > best_metric:  # Identify the best validation metric
            best_metric = val_loss[1]  # Update best validation metric
        # Write loss values to the progress bar for the Epoch. Add best loss values and metrics for comparison
        progress_bar_dict = dict(train_loss=train_loss, val_loss=val_loss[0], best_loss=best_loss, best_IoU=best_metric)
        progress.set_postfix(progress_bar_dict)  # Update the progress bar
    torch.save(model.state_dict(), 'last.pth')  # Save the final state dictionary of the model when training is complete
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()  # Clear GPU cache

import time
import torch
from torch.torch_version import TorchVersion
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import scipy


def compute_color_moments(image_data):
    # Convert the image to a PyTorch tensor
    image_tensor, _ = image_data
    
    # Partition the image into a 10x10 grid
    grid_size = (10, 10)
    channels, height, width  = image_tensor.shape
    
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    
    grid_cells = []

# Divide the image into cells and store them in the list
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell_tensor = image_tensor[:,i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width]
            grid_cells.append(cell_tensor)
    
    feature_vector = []
    
    for cell_tensor in grid_cells:
        moments = []
        
        for channel in range(channels):
                 mean = torch.mean(cell_tensor[channel])
                 std_dev = torch.std(cell_tensor[channel])
                 skewness_value = scipy.stats.skew(cell_tensor[channel].numpy().flatten())  # Using scipy.stats for skewness
                 moments.append((mean.item(), std_dev.item(), skewness_value))              
                 
        feature_vector.append(moments)

    feature_vector = np.array(feature_vector).flatten()
    feature_vector = feature_vector.tolist()
    # print(np.shape(feature_vector))
    return feature_vector

def compute_hog(image_data):
    # Convert the image to grayscale and then to a PyTorch tensor
    image_tensor, _ = image_data
    gray_image = transforms.functional.rgb_to_grayscale(img = image_tensor)
    image_tensor = gray_image.permute(1, 2, 0)[:,:,-1]

    # Partition the image into a 10x10 grid
    grid_size = (10, 10)
    height, width  = image_tensor.shape
    
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    
    grid_cells = []

    filterx = torch.tensor([[-1.0,0.0,1.0]])
    filtery = torch.tensor([[-1.0],[0.0],[1]])
    gradx = []
    grady = []
    hog_features = []
# Divide the image into cells and store them in the list
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            bins = np.zeros(9)
            cell = image_tensor[i * cell_height: (i + 1) * cell_height, j * cell_width: (j + 1) * cell_width]
            grid_cells.append(cell)
            gradx = scipy.signal.convolve2d(cell,filterx, "same")
            grady = scipy.signal.convolve2d(cell,filtery, "same")
            # Calculate Gradient angle and magnitude and update the bins
            for a in range(len(gradx)):
                for b in range(len(gradx[0])):
                    grad_angle = np.arctan2(grady[a][b],gradx[a][b])*(180/np.pi)
                    grad_mag = np.sqrt(np.square(gradx[a][b]+np.square(grady[a][b])))
                    index = int((grad_angle%360)//40)
                    bins[index] += grad_mag
                    
            hog_features.extend(bins)
    
    return hog_features

#Resizing for Resnet Models
def resize_224(image):
    resnet_image = transforms.Resize((224,224))(image)
    resnet_image = resnet_image.unsqueeze(0)
    return resnet_image

def resnet50_avgpool(image_data):
    
    image, _ = image_data
    resnet_image = resize_224(image)
    
    # Load the pre-trained ResNet model
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()
    
    def hook_fn(module, input, output):
        hook_fn.output = output
    
    hook = model.avgpool.register_forward_hook(hook_fn)
    
    # Perform forward pass to compute the 2048-dimensional vector
    model(resnet_image)
    output = hook_fn.output
    # print(hook_fn.output.shape)
    
    # Remove the hook
    hook.remove()

    # Reduce dimensionality to 1024 by averaging two consecutive entries
    reduced_vector = torch.mean(output.view(1024, -1, 2), dim=-1).view(1024)
    
    # print(reduced_vector.shape)
    reduced_vector = np.array(reduced_vector.detach()).flatten()
    reduced_vector = reduced_vector.tolist()
    return reduced_vector  # Return the first target_dim entries

def resnet50_layer3(image_data):
    
    image, _ = image_data
    resnet_image = resize_224(image)
    
    # Load the pre-trained ResNet model
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()
    
    def hook_fn(module, input, output):
        hook_fn.output = output
    
    hook = model.layer3.register_forward_hook(hook_fn)
    
    # Perform forward pass to compute the 2048-dimensional vector
    model(resnet_image)
    output = hook_fn.output
    # print(hook_fn.output.shape)
    
    # Remove the hook
    hook.remove()

    # Reduce dimensionality to 1024 by averaging two consecutive entries
    reduced_vector = torch.mean(output.view(1024, 14, 14), dim=(1,2))
    
    # print(reduced_vector.shape)
    reduced_vector = np.array(reduced_vector.detach()).flatten()
    reduced_vector = reduced_vector.tolist()

    return reduced_vector  # Return the first target_dim entries

def resnet50_fc(image_data):
    
    image, _ = image_data
    resnet_image = resize_224(image)
    
    # Load the pre-trained ResNet model
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    model.eval()
    
    def hook_fn(module, input, output):
        hook_fn.output = output
    
    hook = model.fc.register_forward_hook(hook_fn)
    
    # Perform forward pass to compute the 2048-dimensional vector
    model(resnet_image)
    output = hook_fn.output
    # print(hook_fn.output.shape)
    
    # Remove the hook
    hook.remove()
    
    output = output.squeeze(0)
    output = np.array(output.detach())
    output = output.tolist()

    return output 


def loadDataset():
    dataset_path = "CalTech"
    transform = transforms.Compose([
        transforms.Resize((300, 100)),
        transforms.ToTensor()
    ])
    dataset = datasets.Caltech101(root = dataset_path, transform = transform, download = True)
    return dataset

def checkChannel(image):
    img, _ = image
    channels, _, _ = img.shape
    if channels != 3:
        return True

def main():
    

    # Load the Caltech 101 dataset
    dataset = loadDataset()
    
    for i in range(len(dataset)):

        image,_ = dataset[i]      
        features = []

        # Compute and append all features in the feature list
        features.append(compute_color_moments(image))
        features.append(compute_hog(image))
        features.append(resnet50_avgpool(image))
        features.append(resnet50_layer3(image))
        features.append(resnet50_fc(image))
        print("Image ID: ", i)     
        print("Color Moments: ", features[0])   
        print("HOG: ", features[1])   
        print("Resnet AvgPool: ", features[2])   
        print("Resnet Layer3: ", features[3])   
        print("Resnet Fully Connected: ", features[4])   

if __name__ == "__main__":
    start_tome = time.time()
    main()
    print(f"{time.time() - start_tome} seconds")



# ================================================================== #
#  CNN + NetVLAD for image classification                            #
#                                                                    #
#  Assignment 3 Task 3 | Script by Group 19                          #
# ================================================================== #


# Dependencies
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
import torch.nn.functional as F
plt.style.use('ggplot')


# Image preprocessing
# Resize image to (256, 256) and extract center (224, 224) pixels
# Normalization constants were best parameters chosen for ImageNet dataset
im_trans = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Get datasets
# Within each directory (train/test/validation) ...
# ... there are separate directories for images of every class
train_dir = '../data/train'
valid_dir = '../data/validation'
test_dir = '../data/test'

data = {
    'train': datasets.ImageFolder(root=train_dir, transform=im_trans),
    'valid': datasets.ImageFolder(root=valid_dir, transform=im_trans),
    'test': datasets.ImageFolder(root=test_dir, transform=im_trans)
}

dataloaders = {
    'train': DataLoader(data['train'], batch_size=50, shuffle=True),
    'valid': DataLoader(data['valid'], batch_size=50, shuffle=True),
    'test': DataLoader(data['test'], batch_size=50, shuffle=True)
}


# Define the NetVLAD layer

class NetVLAD(torch.nn.Module):

    def __init__(self, clusters, dim, alpha):

        super(NetVLAD, self).__init__()
        self.clusters = clusters
        self.dim = dim
        self.alpha = alpha
        self.conv = torch.nn.Conv2d(
            dim, clusters, kernel_size=(1, 1), bias=True)
        self.cluster_centers = torch.nn.Parameter(torch.rand(clusters, dim))

        # Initialize weights and biases
        self.init_params()

    def init_params(self):
        # Weights placeholder
        self.conv.weight = torch.nn.Parameter(
            (2.0 * self.alpha * self.cluster_centers).unsqueeze(-1).unsqueeze(-1)
        )
        # Biases placeholder
        self.conv.bias = torch.nn.Parameter(
            - self.alpha * self.cluster_centers.norm(dim=1)
        )

    def forward(self, x):
        """
        Input is a tensor of shape (batch_size, dim, height, width)
        Cluster centers initialized to shape (num_clusters, dim)
        """

        # Soft assignment of weights
        # (1, 1) 2d convolution with num_cluster output channels is performed on the input ...
        # ... which reduces it to shape (batch_size, num_clusters, height, width)
        # Reshaped this tensor to (batch_size, num_clusters, num_desc) ...
        # ... where num_desc = height * width
        desc_weights = self.conv(x).view(x.shape[0], self.clusters, -1)
        # Softmax along num_clusters dimension (1)
        # Final shape = (batch_size, num_clusters, num_desc)
        desc_weights = F.softmax(desc_weights, dim=1)

        # In this section, some matrix manipulations will be performed
        # Final shape for each tensor is mentioned after the command

        x_flat = x.view(x.shape[0], x.shape[1], -1)
        # x_flat -> (batch_size, dim, num_desc)
        resid_x = x_flat.expand(self.clusters, -1, -1, -1).permute(1, 0, 2, 3)
        # resid_x -> (batch_size, num_clusters, dim, num_desc)
        resid_cc = self.cluster_centers.expand(
            x_flat.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # resid_cc -> (batch_size, num_clusters, dim, num_desc)
        resid = (resid_x - resid_cc) * desc_weights.unsqueeze(2)
        # resid -> (batch_size, num_clusters, dim, num_desc)
        vlad = resid.sum(dim=-1)
        # vlad -> (batch_size, num_clusters, dim)

        # Normalize the VLAD features with L2 norm ...
        # ... and reduce to 2D so that next FC layer can accept it
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


# Base classifier model with NetVLAD

model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, kernel_size=(3, 3), stride=1, bias=True),
    torch.nn.ReLU(),
    torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2),
    torch.nn.Conv2d(4, 16, kernel_size=(3, 3), stride=1, bias=True),
    torch.nn.ReLU(),
    NetVLAD(clusters=4, dim=16, alpha=0.5),
    torch.nn.Flatten(),
    torch.nn.Linear(64, 7),
    torch.nn.LogSoftmax(dim=1)
)


# Define train and test functions

def train(model, train_loader, loss_function, optimizer, epoch):

    # Set model in training mode (parameters update)
    model.train()
    # Counter for number of correct predictions
    top_1_correct = 0
    # Counter for target being within top 3 predictions
    top_3_correct = 0
    # Loss collector for every batch
    epoch_loss = []

    for batch_id, (data, target) in enumerate(train_loader):
        # Reset optimizer
        optimizer.zero_grad()
        # Get model output
        output = model(data)
        # Compute average loss for this batch
        loss = loss_function(output, target).mean()
        # Backpropagate loss
        loss.backward()
        # Update optimizer parameters
        optimizer.step()

        # Predicted label
        top_1_preds = output.argmax(dim=1, keepdim=True)
        # Top 3 predicted labels
        top_3_preds = output.topk(3, dim=1)[1]

        top_1_correct += top_1_preds.eq(
            target.view_as(top_1_preds)).sum().item()
        top_3_correct += sum([1 if target[i] in top_3_preds[i] else 0
                              for i in range(len(top_3_preds))])

        # Console output
        print("Epoch {} [Batch {}/{}] \t Average batch loss: {:.4f}".format(
            epoch, batch_id+1, len(train_loader), loss.item()
        )
        )

        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    top_1_accuracy = 100. * top_1_correct / len(train_loader.dataset)
    top_3_accuracy = 100. * top_3_correct / len(train_loader.dataset)

    return avg_epoch_loss, top_1_accuracy, top_3_accuracy


def validate(model, test_loader, loss_function, epoch):

    # Set model in evaluation mode (parameters don't update)
    model.eval()
    # Counter for number of correct predictions
    top_1_correct = 0
    # Counter for target being within top 3 predictions
    top_3_correct = 0
    # Loss collector for every batch
    epoch_loss = []

    for data, target in test_loader:
        # Generate model output
        output = model(data)
        # Compute average loss and update loss history list
        epoch_loss.append(loss_function(output, target).mean().item())

        top_1_preds = output.argmax(dim=1, keepdim=True)
        # Number of correct in top 3 predictions
        top_3_preds = output.topk(3, dim=1)[1]

        # Get number of correct classifications
        top_1_correct += top_1_preds.eq(
            target.view_as(top_1_preds)).sum().item()
        top_3_correct += sum([1 if target[i] in top_3_preds[i] else 0
                              for i in range(len(top_3_preds))])

    test_loss = sum(epoch_loss)/len(epoch_loss)
    top_1_accuracy = 100. * top_1_correct / len(test_loader.dataset)
    top_3_accuracy = 100. * top_3_correct / len(test_loader.dataset)

    print("\nTest Epoch {} \t Top 1 accuracy: {:.2f}% \t Top 3 accuracy: {:.2f}%".format(
        epoch, top_1_accuracy, top_3_accuracy
    ))
    print('\n------------------------------------------------------------------------\n')

    return test_loss, top_1_accuracy, top_3_accuracy


# Main function
epochs = 100
save_interval = 5
loss_function = torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
save_path = '../checkpoints'

train_loss_hist, train_acc1_hist, train_acc3_hist = [], [], []
val_loss_hist, val_acc1_hist, val_acc3_hist = [], [], []


for epoch in range(epochs):
    # Train the model and store loss and accuracies
    train_loss, train_acc1, train_acc3 = train(model, dataloaders['train'],
                                               loss_function, optimizer, epoch)
    # Test the model and store loss and accuracies
    val_loss, val_acc1, val_acc3 = validate(model, dataloaders['valid'],
                                            loss_function, epoch)
    scheduler.step()

    train_loss_hist.append(train_loss)
    train_acc1_hist.append(train_acc1)
    train_acc3_hist.append(train_acc3)
    val_loss_hist.append(val_loss)
    val_acc1_hist.append(val_acc1)
    val_acc3_hist.append(val_acc3)

    if (epoch % save_interval == 0) or (epoch == epochs-1):
        torch.save(model.state_dict(), save_path +
                   '/task3_epoch_{}'.format(epoch))


# Loss history plot
plt.figure(figsize=(6, 6))
plt.plot(train_loss_hist, color='blue', alpha=0.8, linewidth=2, label='Train')
plt.plot(val_loss_hist, color='red', alpha=0.8,
         linewidth=2, label='Validation')
plt.title('Loss history', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Nonlinear logloss')
plt.legend()
plt.savefig('../plots/task3_loss_history.png')
plt.show()

# Top 1 accuracy plot
plt.figure(figsize=(6, 6))
plt.plot(train_acc1_hist, color='blue', alpha=0.8, linewidth=2, label='Train')
plt.plot(val_acc1_hist, color='red', alpha=0.8,
         linewidth=2, label='Validation')
plt.title('Top 1 accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../plots/task3_top1_accuracy.png')
plt.show()

# Top 3 accuracy plot
plt.figure(figsize=(6, 6))
plt.plot(train_acc3_hist, color='blue', alpha=0.8, linewidth=2, label='Train')
plt.plot(val_acc3_hist, color='red', alpha=0.8,
         linewidth=2, label='Validation')
plt.title('Top 3 accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../plots/task3_top3_accuracy.png')
plt.show()

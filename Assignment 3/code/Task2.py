
# ================================================================== #
#  Convolutional NNs for image classification                        #
#                                                                    #
#  Assignment 3 Task 2 | Script by Group 19                          #
# ================================================================== #


# Dependencies
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from matplotlib import pyplot as plt
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


# Define the model
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 4, kernel_size=(3,3), stride=1, bias=True),
    torch.nn.ReLU(),
    torch.nn.AvgPool2d(kernel_size=(2,2), stride=2),
    torch.nn.Conv2d(4, 16, kernel_size=(3,3), stride=1, bias=True),
    torch.nn.ReLU(),
    torch.nn.AvgPool2d(kernel_size=(2,2), stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(46656, 500),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(500, 7),
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

        top_1_correct += top_1_preds.eq(target.view_as(top_1_preds)).sum().item()
        top_3_correct += sum([1 if target[i] in top_3_preds[i] else 0 \
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
        top_1_correct += top_1_preds.eq(target.view_as(top_1_preds)).sum().item()
        top_3_correct += sum([1 if target[i] in top_3_preds[i] else 0 \
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
    train_loss, train_acc1, train_acc3 = train(model, dataloaders['train'], loss_function, optimizer, epoch)
    # Test the model and store loss and accuracies
    val_loss, val_acc1, val_acc3 = validate(model, dataloaders['valid'], loss_function, epoch)
    scheduler.step()

    train_loss_hist.append(train_loss)
    train_acc1_hist.append(train_acc1)
    train_acc3_hist.append(train_acc3)
    val_loss_hist.append(val_loss)
    val_acc1_hist.append(val_acc1)
    val_acc3_hist.append(val_acc3)

    if (epoch % save_interval == 0) or (epoch == epochs-1):
        torch.save(model.state_dict(), save_path+'/task2_epoch_{}'.format(epoch))


# Final statistics
print("\n\n\n\n\n")
print("Train loss: {:.4f}".format(train_loss_hist[-1]))
print("Train accuracy: {:.2f}".format(train_acc1_hist[-1]))
print("Train top-3 accuracy: {:.2f}".format(train_acc3_hist[-1]))
print("Validation loss: {:.4f}".format(val_loss_hist[-1]))
print("Validation accuracy: {:.4f}".format(val_acc1_hist[-1]))
print("Validation top-3 accuracy: {:.4f}".format(val_acc3_hist[-1]))

# Test set performance
test_loss, test_acc1, test_acc3 = validate(model, dataloaders['test'], loss_function, 1)

# Loss history plot
plt.figure(figsize=(6, 6))
plt.plot(train_loss_hist, color='blue', alpha=0.8, linewidth=2, label='Train')
plt.plot(val_loss_hist, color='red', alpha=0.8, linewidth=2, label='Validation')
plt.title('Loss history', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Nonlinear logloss')
plt.legend()
plt.savefig('../plots/task2_loss_history.png')
plt.show()

# Top 1 accuracy plot
plt.figure(figsize=(6, 6))
plt.plot(train_acc1_hist, color='blue', alpha=0.8, linewidth=2, label='Train')
plt.plot(val_acc1_hist, color='red', alpha=0.8, linewidth=2, label='Validation')
plt.title('Top 1 accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../plots/task2_top1_accuracy.png')
plt.show()

# Top 3 accuracy plot
plt.figure(figsize=(6, 6))
plt.plot(train_acc3_hist, color='blue', alpha=0.8, linewidth=2, label='Train')
plt.plot(val_acc3_hist, color='red', alpha=0.8, linewidth=2, label='Validation')
plt.title('Top 3 accuracy', fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../plots/task2_top3_accuracy.png')
plt.show()

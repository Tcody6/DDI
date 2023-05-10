import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tdc.utils import get_label_map
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

labels = get_label_map(name='DrugBank', task='DDI')

x = np.loadtxt("ap_fingerprint.txt", dtype=int)
x_test = torch.tensor(np.loadtxt("ap_fingerprint_test.txt")).double()

N_SAMPLES, D_in, D_out = 134147, 1024, 86

y = np.loadtxt("y_train.txt")

ohe = OneHotEncoder().fit(y.reshape(-1, 1))

y = ohe.transform(y.reshape(-1, 1))


# Define the batch size and the number of epochs
BATCH_SIZE = 64
N_EPOCHS = 3

# Use torch.utils.data to create a DataLoader
# that will take care of creating batches
dataset = TensorDataset(torch.tensor(x).double(), torch.tensor(y.A))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model, loss and optimizer
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, 512).double(),
    torch.nn.ReLU().double(),
    torch.nn.Linear(512, D_out).double(),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Get the dataset size for printing (it is equal to N_SAMPLES)
dataset_size = len(dataloader.dataset)

fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
fig.suptitle('Atom Pair', fontsize=16)
ax[0].set_xlabel('Batch')
ax[0].set_ylabel('Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')

batch_losses = []
accuracies = []

for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch + 1}\n-------------------------------")

    # Loop over batches in an epoch using DataLoader
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):

        y_batch_pred = model(x_batch)

        loss = loss_fn(y_batch_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Every 100 batches, print the loss for this batch
        # as well as the number of examples processed so far
        if id_batch % 100 == 0:
            loss, current = loss.item(), (id_batch + 1) * len(x_batch)
            batch_losses.append(loss)

            print(f"loss: {loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")

        # Calculate accuracy every 10 batches
        if id_batch % 10 == 0:
            with torch.no_grad():
                y_pred = model(x_test)
                labels = y_pred.argmax(axis=1)
                labels = labels + 1
                acc = accuracy_score(labels, np.loadtxt("y_test.txt"))
                accuracies.append(acc)

ax[0].plot(batch_losses)
ax[0].set_xlim(0, len(batch_losses))
ax[0].set_ylim(0, max(batch_losses))
ax[0].set_title('Training Loss')

ax[1].plot(accuracies)
ax[1].set_xlim(0, len(accuracies))
ax[1].set_ylim(0, 1.0)
ax[1].set_title('Accuracy')
plt.show()

# Once training is complete, calculate accuracy on test set

with torch.no_grad():
    output = model(x_test)

output = np.array(output)

labels = output.argmax(axis=1)
labels = labels + 1

acc = accuracy_score(labels, np.loadtxt("y_test.txt"))
print(f"Test set accuracy: {acc:.4f}")
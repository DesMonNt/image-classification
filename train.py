from tkinter import Variable
import torch
from early_stopping import EarlyStopping

def train(model, optimizer, criterion, train_loader, test_loader, num_epochs=10, device='cpu'):
    early_stopping = EarlyStopping()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, real_labels in train_loader:
            images = images.to(device)
            real_labels = real_labels.to(device)

            optimizer.zero_grad()
            predicted_labels = model(images)
            loss = criterion(predicted_labels, real_labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            early_stopping.restore_best_weights(model)
            break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

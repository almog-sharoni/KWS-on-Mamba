import torch
from utils import log_to_file, EarlyStopping
from tqdm import tqdm

def trainig_loop(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler):
    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=6, min_delta=0.001)

    # Training loop
    num_epochs = num_epochs

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # Log new training session
    log_to_file("\n\nNew training session\n\n")
    # Log the model architecture
    log_to_file(str(model))


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for audio, labels in tqdm(train_loader):
            audio, labels = audio.to("cuda"), labels.to("cuda")

            # Forward pass
            outputs = model(audio)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy}%')

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)

        # Log training metrics
        log_to_file(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # # Step the scheduler
        # scheduler.step()

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to("cuda"), labels.to("cuda")
                outputs = model(audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss_avg = val_loss / len(val_loader)
        print(f'Validation Loss: {val_loss_avg}, Validation Accuracy: {val_accuracy}%')

        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss_avg)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss_avg)
        print(f'Learning rate after epoch {epoch+1}: {scheduler.get_last_lr()}')

        
        
        # Check early stopping condition
        if early_stopping.step(val_loss/len(val_loader)):
            print(f"Stopping training at epoch {epoch+1} due to early stopping")
            break

        # Log validation metrics
        log_to_file(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss_avg:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    log_to_file("Training complete.")

    return train_accuracies, val_accuracies, train_losses, val_losses
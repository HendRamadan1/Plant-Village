from tqdm.notebook import tqdm
import torch 
from models.resent_model import model_resent
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
     
    return running_loss/len(dataloader), 100.*correct/total



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name="best_model.pth"):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.*correct/total:.2f}%"})
        
        # --- Validation Phase ---
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        
        # Save History
        history['train_loss'].append(train_loss/len(train_loader))
        history['train_acc'].append(100.*correct/total)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        
        
        print(f"Summary: Train Acc: {history['train_acc'][-1]:.2f}% | Val Acc: {v_acc:.2f}%")
        
        # Save Best Model Checkpoint
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': v_acc,
            }, model_name)
            print(f"🏆 New Best Model Saved!")
            
    return history


from tqdm.notebook import tqdm
from models.resent_model import model_resent
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import torch 


def evaluate_model(model, dataloader, class_names):
    checkpoint = torch.load('/kaggle/working/resnet50_plant_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)


    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n{'='*70}")
    print(f"SUMMARY METRICS")
    print(f"{'='*70}")
    print(f"Accuracy:         {accuracy:.2f}%")
    print(f"Weighted F1:      {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"Macro Avg F1:     {report_dict['macro avg']['f1-score']:.4f}")
    print(f"{'='*70}")


    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



evaluate_model(model_resent, test_loader, FULL_DATASET.classes)

def show_prediction_batch(loader, model, device, class_names):
    checkpoint = torch.load('/kaggle/working/resnet50_plant_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    inputs, labels = next(iter(loader))
    inputs, labels = inputs.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(16, 16))
    
    for i in range(min(16, len(inputs))):
        ax = fig.add_subplot(4, 4, i+1)
        
        img = inputs[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        actual_cls = class_names[labels[i].item()]
        pred_cls = class_names[preds[i].item()]
        color = 'green' if actual_cls == pred_cls else 'red'
        
        ax.set_title(f"True: {actual_cls}\nPred: {pred_cls}", color=color, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

show_prediction_batch(test_loader, model_resent, device, FULL_DATASET.classes)
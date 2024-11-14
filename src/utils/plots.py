
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score

def quantify_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Evaluate the model on training, validation, and test sets
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Print the results
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Collect loss and accuracy values
    loss_values = [train_loss, val_loss, test_loss]
    accuracy_values = [train_accuracy, val_accuracy, test_accuracy]

    # Plotting the results
    plt.figure(figsize=(14, 6))

    # Plot for loss values
    plt.subplot(1, 2, 1)
    plt.bar(['Train', 'Validation', 'Test'], loss_values, color=['skyblue', 'orange', 'green'])
    plt.title('Loss Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Loss')
    plt.ylim([0, max(loss_values) + 0.1])

    # Annotate loss bars
    for i, v in enumerate(loss_values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

    # Plot for accuracy values
    plt.subplot(1, 2, 2)
    plt.bar(['Train', 'Validation', 'Test'], accuracy_values, color=['skyblue', 'orange', 'green'])
    plt.title('Accuracy Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])

    # Annotate accuracy bars
    for i, v in enumerate(accuracy_values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
    return loss_values, accuracy_values

def plot_history(model_history):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['accuracy'], label='Train Accuracy')
    plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], label='Train Loss')
    plt.plot(model_history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

def calculate_metrics(y_true, y_pred_binary):
    """
    Calculate precision, recall, F1 score, and AUC.
    
    Parameters:
        y_true: Ground truth labels.
        y_pred_binary: Predicted labels (binary).
    
    Returns:
        tuple: Precision, Recall, F1-score, AUC
    """
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred_binary)
    return precision, recall, f1, auc

def mean_average_precision(y_true, y_pred_probs):
    """
    Calculate mean average precision (mAP).
    
    Parameters:
        y_true: Ground truth labels.
        y_pred_probs: Predicted probabilities for each sample.
    
    Returns:
        float: mAP score
    """
    return average_precision_score(y_true, y_pred_probs)

def calculate_iou(y_true, y_pred_binary):
    """
    Calculate Intersection over Union (IoU) for segmentation tasks.

    Parameters:
        y_true: Ground truth labels.
        y_pred_binary: Predicted labels (binary).

    Returns:
        float: IoU score
    """
    intersection = np.sum(np.logical_and(y_true == 1, y_pred_binary == 1))
    union = np.sum(np.logical_or(y_true == 1, y_pred_binary == 1))
    iou = intersection / float(union) if union != 0 else 0
    return iou

def evaluate_and_plot(model, X_train, y_train, X_val, y_val, X_test, y_test, threshold = 0.5):
    """
    Evaluates the model on the training, validation, and test datasets,
    and plots the loss and accuracy for each. Also reports additional metrics like 
    precision, recall, f1-score, AUC, mAP, confusion matrix, and IOU.

    Parameters:
        model: Trained model to be evaluated.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        X_test, y_test: Test data and labels.
    """
    # Evaluate the model on training, validation, and test sets
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Get the predicted probabilities for the test data
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Convert probabilities to binary labels (using a threshold of 0.5)
    y_train_pred_binary = (y_train_pred > threshold).astype(int)
    y_val_pred_binary = (y_val_pred > threshold).astype(int)
    y_test_pred_binary = (y_test_pred > threshold).astype(int)

    # Calculate additional performance metrics
    train_precision, train_recall, train_f1, train_auc = calculate_metrics(y_train, y_train_pred_binary)
    val_precision, val_recall, val_f1, val_auc = calculate_metrics(y_val, y_val_pred_binary)
    test_precision, test_recall, test_f1, test_auc = calculate_metrics(y_test, y_test_pred_binary)

    # Calculate mAP score for each dataset
    train_map = mean_average_precision(y_train, y_train_pred)
    val_map = mean_average_precision(y_val, y_val_pred)
    test_map = mean_average_precision(y_test, y_test_pred)

    # Calculate IoU for train, validation, and test sets
    train_iou = calculate_iou(y_train, y_train_pred_binary)
    val_iou = calculate_iou(y_val, y_val_pred_binary)
    test_iou = calculate_iou(y_test, y_test_pred_binary)

    # Create a dataframe to display the metrics in table format
    metrics = {
        "Dataset": ["Training", "Validation", "Test"],
        "Loss": [train_loss, val_loss, test_loss],
        "Accuracy": [train_accuracy, val_accuracy, test_accuracy],
        "Precision": [train_precision, val_precision, test_precision],
        "Recall": [train_recall, val_recall, test_recall],
        "F1-Score": [train_f1, val_f1, test_f1],
        "AUC": [train_auc, val_auc, test_auc],
        "mAP": [train_map, val_map, test_map],
        "IoU": [train_iou, val_iou, test_iou]  # Adding IoU to the metrics table
    }
    
    df_metrics = pd.DataFrame(metrics)
    print(df_metrics)
    
    # Find best IoU across all datasets
    best_iou = max(train_iou, val_iou, test_iou)
    print(f"Best IoU: {best_iou:.4f}")
    
    # Plotting the results
    loss_values = [train_loss, val_loss, test_loss]
    accuracy_values = [train_accuracy, val_accuracy, test_accuracy]

    # Plot for loss values
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(['Train', 'Validation', 'Test'], loss_values, color=['skyblue', 'orange', 'green'])
    plt.title('Loss Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Loss')
    plt.ylim([0, max(loss_values) + 0.1])

    # Annotate loss bars
    for i, v in enumerate(loss_values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

    # Plot for accuracy values
    plt.subplot(1, 2, 2)
    plt.bar(['Train', 'Validation', 'Test'], accuracy_values, color=['skyblue', 'orange', 'green'])
    plt.title('Accuracy Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])

    # Annotate accuracy bars
    for i, v in enumerate(accuracy_values):
        plt.text(i, v + 0.02, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Confusion Matrix Function
    def plot_confusion_matrix(y_true, y_pred_binary, dataset_name):
        cm = confusion_matrix(y_true, y_pred_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for {dataset_name} Dataset')
        plt.show()

    # Plot confusion matrices
    plot_confusion_matrix(y_train, y_train_pred_binary, "Training")
    plot_confusion_matrix(y_val, y_val_pred_binary, "Validation")
    plot_confusion_matrix(y_test, y_test_pred_binary, "Test")

    # AUC Curve Plotting
    def plot_roc_curve(fpr, tpr, auc_score, label):
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc_score:.2f})')

    # ROC Curve for training, validation, and test sets
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

    train_auc = auc(fpr_train, tpr_train)
    val_auc = auc(fpr_val, tpr_val)
    test_auc = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr_train, tpr_train, train_auc, 'Training')
    plot_roc_curve(fpr_val, tpr_val, val_auc, 'Validation')
    plot_roc_curve(fpr_test, tpr_test, test_auc, 'Test')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random classifier line
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()



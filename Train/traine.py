import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from SwinT import SwinTClassifier


def train(train_loader, swin_type, dataset, epochs, model, lf, token_num,
          optimizer, criterion, device, show_per, reg_type=None, reg_lambda=0., validation=None):
    model.train()
    total_batch = len(train_loader)
    train_test_hist = []
    best_test_acc = -99
    training_times = []


    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    specific_dir = f'./SavedModel/{dataset}/SparseSwin_reg_{reg_type}_lbd_{reg_lambda}_lf_{lf}_{token_num}_{current_time}'
    os.makedirs(specific_dir, exist_ok=True)
    print(f"目录 {specific_dir} 已存在或创建成功。")
    print(os.getcwd())


    plots_dir = os.path.join(specific_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"[TRAIN] 总批次：{total_batch} | 类型：{swin_type} | 正则化：{reg_type} 正则系数：{reg_lambda}")
    total_start_time = time.time()

    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []

    best_epoch = 0
    best_test_acc = -99

    for epoch in range(epochs):
        start_time = time.time()
        epoch_timestamp = time.time()
        print(f"第Epoch {epoch + 1}/{epochs}轮训练开始")
        running_loss, n_correct, n_sample = 0.0, 0.0, 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if isinstance(model, SwinTClassifier) or swin_type.lower() in ["swin_transformer_tiny",
                                                                           "swin_transformer_small",
                                                                           "swin_transformer_base"]:
                outputs = model(inputs)
            else:
                outputs, attn_weights, sparse_token_dim_converter = model(inputs)
                reg = 0
                if reg_type == 'l1':
                    reg = sum(torch.sum(torch.abs(attn_w)) for attn_w in attn_weights)
                elif reg_type == 'l2':
                    reg = sum(torch.sum(attn_w ** 2) for attn_w in attn_weights)
                reg = reg_lambda * reg
                loss = criterion(outputs, labels) + reg
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            with torch.no_grad():
                n_correct_per_batch = torch.sum(torch.argmax(outputs, dim=1) == labels)
                n_correct += n_correct_per_batch
                n_sample += labels.shape[0]
                acc = n_correct / n_sample

            if ((i + 1) % show_per == 0) or ((i + 1) == total_batch):
                print(f'  [{i + 1}/{total_batch}] Loss: {(running_loss / (i + 1)):.4f} Acc : {acc:.4f}')

        epoch_duration = time.time() - start_time

        epoch_time = time.time() - start_time
        training_times.append(epoch_time)
        print(f"第 {epoch + 1} 轮训练时长：{epoch_duration:.2f} 秒")


        test_loss, test_acc, inference_time = test(validation, swin_type=swin_type, model=model, criterion=criterion,
                                                   device=device, plots_dir=plots_dir)
        train_loss, train_acc = (running_loss / total_batch), (n_correct / n_sample)



        train_accs.append(train_acc.item())
        test_accs.append(test_acc)

        train_losses.append(train_loss)
        test_losses.append(test_loss)


        test_loss, train_loss = round(test_loss, 4), round(train_loss, 4)
        train_test_hist.append([train_loss, round(train_acc.item(), 4), test_loss, round(test_acc.item(), 4)])



        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f'{specific_dir}/model_best_epoch_{epoch + 1}_{epoch_timestamp}.pt')
            print(f"模型已保存，epoch {epoch + 1} 准确率：{test_acc:.4f}")


    plot_accuracy(train_accs, test_accs, plots_dir)

    plot_loss(train_losses, test_losses, plots_dir)

    plot_accuracy_with_time(train_accs, test_accs, training_times, plots_dir)

    total_training_time = time.time() - total_start_time
    print(f"总训练时长：{total_training_time / 60:.2f} 分钟")

    results = {
        "Model": swin_type,
        "Best Test Accuracy": best_test_acc,
        "Best Epoch": best_epoch,
        "Inference Time (ms/sample)": inference_time
    }
    df = pd.DataFrame([results])
    df.to_csv(os.path.join(specific_dir, "experiment_results.csv"), index=False)
    print(f"实验结果已保存至 {specific_dir}/experiment_results.csv")


def test(val_loader, swin_type, model, criterion, device, plots_dir,
         class_names=["glioma", "meningioma", "notumor", "pituitary"]):
    model.eval()



    with torch.no_grad():
        total_batch = val_loader.__len__()
        print(f"[TEST] Total : {total_batch} | type : {swin_type}")
        running_loss, n_correct, n_sample = 0.0, 0.0, 0.0
        all_preds = []
        all_labels = []

        total_inference_time = 0
        total_samples = 0

        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()

            if isinstance(model, SwinTClassifier) or swin_type.lower() in ["swin_transformer_tiny",
                                                                           "swin_transformer_small",
                                                                           "swin_transformer_base"]:
                outputs = model(inputs)

            else:
                outputs, attn_weights, sparse_token_dim_converter = model(inputs)  # 对于其他模型类型，获取输出和注意力权重

            end_time = time.time()
            inference_time = end_time - start_time

            total_inference_time += inference_time
            total_samples += inputs.shape[0]

            loss = criterion(outputs, labels)


            running_loss += loss.item()

            n_correct_per_batch = torch.sum(torch.argmax(outputs, dim=1) == labels)
            n_correct += n_correct_per_batch
            n_sample += labels.shape[0]


            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            acc = n_correct / n_sample
            avg_loss = running_loss / total_batch

    avg_inference_time_per_sample = (total_inference_time / total_samples) * 1000

    acc = n_correct / n_sample
    avg_loss = running_loss / total_batch


    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, plots_dir)


    plot_metrics(all_labels, all_preds, class_names, plots_dir)

    print(
        f'[Model : {swin_type}] Loss: {(avg_loss):.4f} Acc : {acc:.4f}')
    return avg_loss, acc, avg_inference_time_per_sample



def plot_confusion_matrix(cm, plots_dir):

    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


    cm_accuracy = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_accuracy = np.nan_to_num(cm_accuracy)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_accuracy, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Accuracy')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.show()


def plot_metrics(all_labels, all_preds, class_names, plots_dir):

    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)


    metrics = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


    metrics_melted = metrics.melt(id_vars="Class", var_name="Metric", value_name="Score")


    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="Class", y="Score", hue="Metric", data=metrics_melted, ax=ax, palette="viridis")


    plt.title("Metrics Comparison Across Classes", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.ylim(0, 1.1)  # 设置 y 轴范围为 [0, 1.1]


    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=False)


    plt.tight_layout()


    plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'))
    plt.close()


def plot_accuracy(train_accs, test_accs, plots_dir):

    train_accs = np.array([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accs])
    test_accs = np.array([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in test_accs])


    plt.figure(figsize=(8, 6))  # 确保是独立的 Figure
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy")
    plt.plot(range(1, len(test_accs) + 1), test_accs, label="Test Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Testing Accuracy')


    plt.savefig(f"{plots_dir}/accuracy_plot.png")
    plt.close()


def plot_loss(train_losses, test_losses, plots_dir):

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss', color='blue')
    plt.plot(test_losses, label='Test loss', color='orange')
    plt.title('Loss curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'))
    plt.close()


def plot_inference_time(models, times, plots_dir):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=times, palette="coolwarm")
    plt.title("Inference Time Comparison (ms/sample)")
    plt.ylabel("Inference Time (ms)")
    plt.xlabel("SparseSwinSDT")
    plt.ylim(0, max(times) * 1.2)
    plt.savefig(os.path.join(plots_dir, "inference_time_comparison.png"))
    plt.show()
    plt.close()

def plot_accuracy_with_time(train_accs, val_accs, training_times, plots_dir):

        train_accs = np.array([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accs])
        val_accs = np.array([acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_accs])
        training_times = np.array([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in training_times])


        cumulative_training_times = np.cumsum(training_times)


        fig, ax1 = plt.subplots(figsize=(10, 6))


        l1, = ax1.plot(range(1, len(train_accs) + 1), train_accs, label="Train Accuracy", color='tab:blue')
        l2, = ax1.plot(range(1, len(val_accs) + 1), val_accs, label="Validation Accuracy", color='tab:orange')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')


        ax2 = ax1.twinx()
        l3, = ax2.plot(range(1, len(cumulative_training_times) + 1), cumulative_training_times,
                       label="Cumulative Training Time (s)", color='tab:green')
        ax2.set_ylabel('Cumulative Time (seconds)')


        final_time = cumulative_training_times[-1]  # 获取最终训练时间
        ax2.text(len(cumulative_training_times), final_time, f'{final_time:.2f} s',
                 color='tab:green', fontsize=12, verticalalignment='bottom', horizontalalignment='right')


        ax1.legend(handles=[l1, l2, l3], loc='lower right', bbox_to_anchor=(0.98, 0.02), frameon=True)


        ax1.set_title('Training and Testing Accuracy with Cumulative Training Time')


        plt.tight_layout()
        plt.savefig(f"{plots_dir}/accuracy_and_cumulative_time_plot.png")
        plt.close()

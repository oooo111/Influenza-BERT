import os
import torch
import pandas as pd
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class VirusDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        sequence = ''.join(c for c in sequence.upper() if c in ['A', 'T', 'G', 'C', 'N'])
        
        encoding = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_test_data(fasta_file, label_file, label_mapping_file):
    label_to_idx = {}
    idx_to_label = {}
    with open(label_mapping_file, 'r') as f:
        for line in f:
            idx, label = line.strip().split('\t')
            idx = int(idx)
            label_to_idx[label] = idx
            idx_to_label[idx] = label
    
    # 加载序列
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        main_id = record.id.split('|')[0]
        seq_ids.append(main_id)
    
    print(f"加载了 {len(sequences)} 条测试序列")
    
    # 加载标签
    labels_df = pd.read_csv(label_file)
    print(f"测试标签文件包含 {len(labels_df)} 条记录")
    id_to_label = dict(zip(labels_df['accession'], labels_df['subtype']))
    
    # 匹配序列和标签
    labels = []
    valid_seq_ids = []
    valid_sequences = []
    
    for i, seq_id in enumerate(seq_ids):
        if seq_id in id_to_label:
            label = id_to_label[seq_id]
            if label in label_to_idx: 
                labels.append(label_to_idx[label])
                valid_seq_ids.append(seq_id)
                valid_sequences.append(sequences[i])
            else:
                print(f"警告: 标签 {label} 在训练集中未出现，跳过序列 {seq_id}")
        else:
            print(f"警告: 序列ID {seq_id} 在标签文件中未找到")
    
    print(f"成功匹配了 {len(labels)} 条测试序列的标签")
    
    return valid_sequences, labels, valid_seq_ids, idx_to_label

def plot_confusion_matrix(y_true, y_pred, idx_to_label, output_dir):
    from sklearn.metrics import confusion_matrix



    cm = confusion_matrix(y_true, y_pred)
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]

    # 统一外部字体大小
    fontsize = 38

    plt.figure(figsize=(len(labels) * 1.8, len(labels) * 1.5), dpi=800)  # 放大格子
    ax = sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels,
    annot_kws={"size": 25}  # 数字更小
        )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize)

    # colorbar 字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)

    plt.xlabel('Predicted Label', fontsize=fontsize)
    plt.ylabel('True Label', fontsize=fontsize)
    plt.title('Confusion Matrix', fontsize=fontsize)

    ax.tick_params(axis='both', which="major", labelsize=fontsize)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ['png', 'svg', 'pdf']:
        path = os.path.join(output_dir, f'segment_matrix.{ext}')
        plt.savefig(path, dpi=800, bbox_inches='tight')
        print(f"混淆矩阵已保存为 {path}")

    plt.close()


def evaluate_model():
    # 设置参数
    
    model_dir = "your_model"     # output_512_mlm随机初始化embedding
    test_fasta_file = "test.fasta"
    test_label_file = "test_labels.csv"
    label_mapping_file = os.path.join(model_dir, "label_mapping.txt")
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载测试数据
    test_sequences, test_labels, test_ids, idx_to_label = load_test_data(
        test_fasta_file, test_label_file, label_mapping_file
    )
    
    # 加载模型和tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device)
    model.eval()
    
    # 创建测试数据集和数据加载器
    test_dataset = VirusDataset(test_sequences, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 进行预测
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    labels = np.unique(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print("\n每个类别指标:")
    for i, label in enumerate(labels):
        # 单个类别的准确率 = 该类别预测正确的数量 / 该类别真实样本总数
        idx = np.array(all_labels) == label
        class_acc = np.sum(np.array(all_preds)[idx] == label) / np.sum(idx)
        print(f"类别 {idx_to_label[label]}: Accuracy={class_acc:.4f}")

    
    print("\n测试集评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 生成详细的分类报告
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    class_report = classification_report(all_labels, all_preds, target_names=target_names)
    print("\n分类报告:")
    print(class_report)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, idx_to_label, model_dir)
    print(f"混淆矩阵已保存到 {os.path.join(model_dir, 'confusion_matrix.png')}")
    
    # 保存详细的预测结果
    results_df = pd.DataFrame({
        'sequence_id': test_ids,
        'true_label': [idx_to_label[label] for label in all_labels],
        'predicted_label': [idx_to_label[pred] for pred in all_preds],
        'correct': [pred == label for pred, label in zip(all_preds, all_labels)]
    })
    
    results_file = os.path.join(model_dir, 'test_predictions.csv')
    results_df.to_csv(results_file, index=False)
    print(f"详细预测结果已保存到 {results_file}")

if __name__ == "__main__":
    evaluate_model()
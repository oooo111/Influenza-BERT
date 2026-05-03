import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from Bio import SeqIO
from transformers import AutoTokenizer, AutoConfig
from transformers import BertConfig


# 1. 数据加载和预处理
class VirusDataset(Dataset):
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

def load_data(fasta_file, label_file, existing_label_to_idx=None):
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
        main_id = record.id.split('|')[0]
        seq_ids.append(main_id)
    
    print(f"加载了 {len(sequences)} 条序列")
    
    labels_df = pd.read_csv(label_file)
    print(f"标签文件包含 {len(labels_df)} 条记录")
    id_to_label = dict(zip(labels_df['accession'], labels_df['virus_type']))
    
    # 获取标签列表
    labels = []
    valid_seq_ids = []
    valid_sequences = []
    
    for i, seq_id in enumerate(seq_ids):
        if seq_id in id_to_label:
            labels.append(id_to_label[seq_id])
            valid_seq_ids.append(seq_id)
            valid_sequences.append(sequences[i])
        else:
            print(f"警告: 序列ID {seq_id} 在标签文件中未找到")
    
    print(f"成功匹配了 {len(labels)} 条序列的标签")
    
    # 获取唯一标签并映射到数字
    if existing_label_to_idx is None:
        unique_labels = sorted(set(labels))
        print(f"共有 {len(unique_labels)} 个不同的标签类别: {unique_labels}")
        
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    else:
        label_to_idx = existing_label_to_idx
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        print(f"使用预定义的标签映射，共 {len(label_to_idx)} 个类别")
    
    # 将标签转换为数字
    numeric_labels = [label_to_idx[label] for label in labels]
    
    return valid_sequences, numeric_labels, label_to_idx, idx_to_label

def compute_metrics(pred):
    labels = pred.label_ids
    
    # 处理predictions是元组的情况
    if isinstance(pred.predictions, tuple):
        # 通常第一个元素是logits
        preds = pred.predictions[0].argmax(-1)
    else:
        preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    train_fasta_file = ""
    train_label_file = ""
    val_fasta_file = ""
    val_label_file = ""
    model_name = ""   #huggingface download
    output_dir = ""
    batch_size = 4  
    epochs = 10
    
    # 加载训练数据
    train_sequences, train_labels, label_to_idx, idx_to_label = load_data(train_fasta_file, train_label_file)
    num_labels = len(label_to_idx)
    
    # 加载验证数据
    val_sequences, val_labels, _, _ = load_data(val_fasta_file, val_label_file, label_to_idx)
    
    print(f"训练集大小: {len(train_sequences)}, 验证集大小: {len(val_sequences)}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)  
    config.num_labels = num_labels  
    config.problem_type = "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True
    )
    
    # 创建数据集
    train_dataset = VirusDataset(train_sequences, train_labels, tokenizer)
    val_dataset = VirusDataset(val_sequences, val_labels, tokenizer)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,   
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        # 禁用混合精度训练
        fp16=True,
        bf16=False,
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 训练模型
    print("开始训练模型...")
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(output_dir)
    
    # 保存标签映射
    with open(os.path.join(output_dir, "label_mapping.txt"), "w") as f:
        for idx, label in idx_to_label.items():
            f.write(f"{idx}\t{label}\n")
    
    # 评估训练集
    print("\n训练集评估结果:")
    train_results = trainer.evaluate(train_dataset)
    for key, value in train_results.items():
        print(f"{key}: {value:.4f}")
    
    # 评估验证集
    print("\n验证集评估结果:")
    eval_results = trainer.evaluate(val_dataset)
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\n模型已保存到 {output_dir}")
    print(f"标签映射已保存到 {os.path.join(output_dir, 'label_mapping.txt')}")

if __name__ == "__main__":
    main()
# Influenza-BERT
This project primarily addresses the challenge of identifying influenza virus subtypes under long-tailed data distribution.

## Pre-Training
Our pre-trained models have already been uploaded to Hugging Face(https://huggingface.co/rongye1/Influenza_BERT/tree/main/Pre-training). 

## Finetune
The user needs to provide a FASTA file containing the sequences and a CSV file, with the data format as shown in the test_data folder. Then, use the subtype_finetune.ipynb file.
Our 10 subtype pre-trained models have already been uploaded to Hugging Face(https://huggingface.co/rongye1/Influenza_BERT/tree/main/checkpoint). We will later upload pre-trained models for classification of all subtypes in the database.

## Eval
The eval.ipynb file is a script for evaluating the test set.

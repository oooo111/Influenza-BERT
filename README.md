# Influenza-BERT
This project primarily addresses the challenge of identifying influenza virus subtypes under long-tailed data distribution.

## Pre-Training
Our pre-trained models have already been uploaded to Hugging Face(https://huggingface.co/rongye1/Influenza_BERT/tree/main/Pretraining_1024_mlm). 

## Subtype_Task_Finetune
The user needs to provide a FASTA file containing the sequences and a CSV file, with the data format as shown in the test_data folder. Then, use the subtype_finetune.ipynb file.
For an input of any single sequence fragment, our 10 subtype fine-tuned models have been uploaded to Hugging Face.(https://huggingface.co/rongye1/Influenza_BERT/tree/main/any_segment_subtype_10).

For input HA/NA fragments, our 10 subtype fine-tuned models have been uploaded to Hugging Face.(https://huggingface.co/rongye1/Influenza_BERT/tree/main/HA_NA_subtype10)

## Pathogenicity prediction
200-shot fine-tuned model for pathogenicity prediction. (https://huggingface.co/rongye1/Influenza_BERT/tree/main/200shot_InfulBERT_model)

## Eval
The eval.ipynb file is a script for evaluating the test set.


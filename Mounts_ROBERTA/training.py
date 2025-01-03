import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaForTokenClassification, Trainer, TrainingArguments, RobertaTokenizerFast, DataCollatorForTokenClassification
import torch.nn.functional as F
import ast
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

# Initialize the tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)

# Function to read and process CSV data
def load_csv(data_type="train"):
    """
    Load and preprocess the CSV dataset.
    
    Args:
        data_type (str): Type of dataset ('train', 'val', or 'test').
        
    Returns:
        pd.DataFrame: Processed dataframe containing tokens and labels.
    """
    df = pd.read_csv(f"datasets/{data_type}_processed.csv", index_col=0)
    df['tokens'] = df["tokens"].apply(ast.literal_eval)  # Convert tokens from string to list of tokens
    df['labels'] = df["labels"].apply(ast.literal_eval)  # Convert labels from string to list of labels
    return df

# Custom Dataset class for NER (Named Entity Recognition)
class NERDataset(Dataset):
    def __init__(self, df, max_length=128):
        """
        Initialize the NER dataset.
        
        Args:
            df (pd.DataFrame): Dataframe containing tokenized text and labels.
            max_length (int): Maximum length of tokenized input sequences.
        """
        self.df = df
        self.max_length = max_length
        self.df = self._tokenize_and_align_labels(self.df)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single data sample.

        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        return {
            'input_ids': self.df.loc[idx, "input_ids"],
            'attention_mask': self.df.loc[idx, "attention_mask"],
            'labels': self.df.loc[idx, "labels"]
        }
    
    # Tokenize and align NER labels with subword tokens
    def tokenize_and_align(self, example):
        """
        Tokenize and align NER labels with subword tokens.
        
        Args:
            example (dict): A dictionary containing tokens and labels.
            
        Returns:
            dict: Tokenized inputs with aligned labels.
        """
        tokenized_inputs = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        word_ids = tokenized_inputs.word_ids()  # Map each token to its original word
        aligned_labels = []
        prev_word_id = None
        cur_idx = 0
        for idx, word_id in enumerate(word_ids):
            if word_id is None:  # Special tokens like [CLS], [SEP]
                aligned_labels.append(-100)  # Ignore labels for special tokens
            elif word_id != prev_word_id:
                aligned_labels.append(example["labels"][cur_idx])
                prev_word_id = word_id
                cur_idx += 1
            else:
                aligned_labels.append(-100)  # Subword tokens: ignore the label
        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs
    
    def _tokenize_and_align_labels(self, dataset):
        """
        Apply tokenization and label alignment to the entire dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to tokenize and align.
            
        Returns:
            pd.DataFrame: Tokenized dataset with aligned labels.
        """
        ds = pd.DataFrame(columns=["input_ids", "attention_mask", "labels"])
        for i in range(len(dataset)):
            ds.loc[i] = self.tokenize_and_align(self.df.loc[i])
        return ds

# Focal Loss Trainer class extending Trainer from Hugging Face
class FocalLossTrainer(Trainer):
    def __init__(self, *args, gamma=2.0, alpha=None, **kwargs):
        """
        Initialize the Focal Loss Trainer.
        
        Args:
            *args: Arguments passed to the parent Trainer class.
            gamma (float): Focal loss focusing parameter to control how much emphasis is placed on hard-to-classify examples.
            alpha (torch.Tensor or None): Optional class weighting for focal loss.
        """
        self.gamma = gamma
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the focal loss.
        
        Args:
            model (nn.Module): The model.
            inputs (dict): Input data.
            return_outputs (bool): Whether to return outputs.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def focal_loss(self, logits, labels):
        """
        Compute the focal loss.
        
        Args:
            logits (torch.Tensor): Model predictions.
            labels (torch.Tensor): Ground truth labels.
        
        Returns:
            torch.Tensor: Focal loss value.
        """
        ce_loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), labels.view(-1), reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, labels.view(-1))
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Function to compute metrics for NER evaluation
def compute_metrics(predictions_and_labels):
    """
    Compute NER evaluation metrics: precision, recall, F1, and accuracy.
    
    Args:
        predictions_and_labels (tuple): Tuple containing predictions and true labels.
        
    Returns:
        dict: Dictionary containing computed metrics.
    """
    predictions, labels = predictions_and_labels
    predicted_labels = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predicted_labels, labels):
        for pred_label, true_label in zip(pred_seq, label_seq):
            if true_label != -100:  # Ignore padding and special tokens
                true_labels.append(true_label)
                pred_labels.append(pred_label)
    
    # Calculate precision, recall, F1-score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None, labels=[0, 1])
    macro_f1 = np.mean(f1)  # Compute macro-average F1 score
    accuracy = accuracy_score(true_labels, pred_labels)  # Compute accuracy
    
    # Return the calculated metrics
    return {
        "precision 0": precision[0],  # Precision for class 0
        "recall 0": recall[0],        # Recall for class 0
        "f1 0": f1[0],                # F1 score for class 0
        "precision 1": precision[1],  # Precision for class 1
        "recall 1": recall[1],        # Recall for class 1
        "f1 1": f1[1],                # F1 score for class 1
        "macro_f1": macro_f1,         # Macro-average F1 score
        "accuracy": accuracy          # Accuracy
    }

if __name__ == "__main__":
    # Load the train, validation, and test datasets
    train_df = load_csv("train")
    val_df = load_csv("val")
    test_df = load_csv("test")
    
    # Create datasets for training, validation, and testing
    train_dataset = NERDataset(train_df)
    valid_dataset = NERDataset(val_df)
    test_dataset = NERDataset(test_df)
    
    # Initialize the data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Load the pre-trained RoBERTa model for token classification
    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU if available
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",            # Directory to save results
        eval_strategy="epoch",             # Evaluate after each epoch
        learning_rate=5e-5,                # Learning rate
        per_device_train_batch_size=16,   # Batch size for training
        per_device_eval_batch_size=16,    # Batch size for evaluation
        num_train_epochs=10,                # Number of training epochs
        weight_decay=0.01                  # L2 regularization coefficient
    )
    
    # Initialize the Focal Loss Trainer
    trainer = FocalLossTrainer(
        model=model,                      # The pre-trained RoBERTa model with token classification head.
        args=training_args,               # Training arguments specifying parameters like output directory, batch size, learning rate, etc.
        train_dataset=train_dataset,      # The training dataset containing tokenized inputs and labels.
        eval_dataset=valid_dataset,       # The validation dataset for evaluation during training.
        data_collator=data_collator,     # A collator to handle padding, tokenization, and batch creation.
        tokenizer=tokenizer,              # The tokenizer used to tokenize the input data.
        compute_metrics=compute_metrics   # The function to compute evaluation metrics like precision, recall, F1 score, and accuracy.
)
    
    # Train the model
    trainer.train()
    
    # Evaluate on the test dataset
    evaluation_results = trainer.evaluate(test_dataset)
    
    # Print evaluation results
    print(evaluation_results)

    # Save the trained model and tokenizer
    model.save_pretrained('./saved_ROBERTA')
    tokenizer.save_pretrained('./saved_ROBERTA')

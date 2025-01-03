# Mountain Named Entity Recognition (NER) Model

## Project Overview
This project focuses on fine-tuning a RoBERTa-based model (roberta-base) to enhance Named Entity Recognition (NER) for mountain names in text. The model has been designed to accurately identify mountain mentions and distinguish them from other geographic entities or non-entities.
You can find my model in HuggingFace [models][https://huggingface.co/UkrKreuzritter/NER_mountain/tree/main]
### Features:
- Fine-tuned on the [mountains NER dataset from Hugging Face](https://huggingface.co/datasets/telord/mountains-ner-dataset/viewer), which contains both mountain and non-mountain sentences.  
- Utilizes **focal loss** to address class imbalance, ensuring the model prioritizes the accurate classification of rare mountain names.  
- Employs token-level classification to distinguish between "mountain" and "non-mountain" entities.  


## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mountain-ner
    cd mountain-ner
    ```
2. Install dependencies: Ensure that you have Python 3.8 or later installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```



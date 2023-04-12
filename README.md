# CSI5180-VA-Project
Question Answering on SQuAD Dataset using different Transformer and attention based models 

## Results
Dataset: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)

Trained on 10 epochs
| Model | Accuracy (%) (Top-1, validation) | F1-Score (%) (Top-1, validation) |
|--------------|:----------:|:----------:|
| BERT | 7.1 | 7.1 | 
| XLNet | 47.2 | 47.2 |

## Development Environment
- OS: Ubuntu 20.04 (64bit)
- GPU: Nvidia Tesla T4
- Language: Python 3.9.6
- Pytorch: 2.0.0

## Requirements

Please install the following library requirements specified in the **requirements.txt** first.

    torch==2.0.0
    torchmetrics==0.11.0
    transformers==4.24.0
    datasets==2.7.0

## Execution

To train BERT on SQuAD for 10 epochs
- Create directory 'model_weights/BERT' in the current directory
- Replace the 'save_path' in 'bert.py'
- Then execute 'bert.py'

      python bert.py
- Follow the same steps for XLNet in 'xlnet.py'
- For evaluation, replace the 'load_path' with the path where model_weights are saved from training.
- Then execute 'bert_evaluate.py' or 'xlnet_evaluate.py'
      
      python bert_evaluate.py
 
 - For inference, replace the 'load_path' and execute the cells in 'bert_inference.ipynb' (same applies for XLNet).
 
 ## Pre-trained Weights (ours; 10 epochs; not SOTA)
 
 Here is the link for pre-trained weights we trained for course of this project of CSI 5180 - Topics in AI: Virtual Assistants.
 
 weights: [here](https://drive.google.com/drive/folders/16-IxLWkIn9zqk5yQLtsfAP2m0yvhLxLP?usp=share_link)
 
 ## Thank you!

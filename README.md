# 🔬 ML/DL Experiment Application


### 🗂️ Application Structure
This Project implement with Pytorch. The overview structure is as follows.
```
Experiment Application
│
├── experiment
│	├── losses
│	│	├── loss.py
│	│	└── distance_metric.py  # for metric learning
│	├── metrics
│	│	└── metric.py
│	├── models
│	│	├── attention
│	│	│    ├── transformer.py
│	│	│    ├── vit.py  # vision transformer
│	│	│    ├── deberta.py
│	│	│    ├── electra.py
│	│	│    ├── spanbert.py
│	│	│    └── gpt2.py
│	│	├── recurrent
│	│	│    ├── rnn.py
│	│	│    ├── lstm.py  
│	│	│    └── gru.py
│	│	├── convolution
│	│	└── probability
│	├── pooling
│	│
│	├── tokenizer
│	│
│	└── tuner
│	    ├── mlm.py  # Masked Language Model (Sub Word Masking/Whole Word Masking) 
│	    ├── clm.py  # Causal Language Model
│	    ├── rtd.py  # Replaced Token Detection
│	    ├── sbo.py  # Span Boundary Objective
│	    └── p_tuning.py 
│
├── dataset_class
│   ├── data_folder  # input your dataset
│   ├── dataclass.py
│   └── preprocessing.py
│  
├── model
│   ├── abstract_task.py
│   ├── model.py
│   └── model_utils.py
│
└── trainer
    ├── train_loop.py
    ├── trainer.py
    └── trainer_utils.py    
```

### Attention
- **[Transformer] Attention Is All You Need (Complete, [Review](https://qcqced123.github.io/nlp/transformer))**

- [Longformer] Longformer: The Long-Document Transformer (Continue)

- [Reformer] Reformer: The Efficient Transformer (Continue)

- **[ELECTRA] Pre-training Text Encoders as Discriminators Rather Than Generators (Complete)**

- **[SpanBERT] SpanBERT: Improving Pre-training by Representing and Predicting Spans (Complete)**

- [DistilBERT] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

- [Sentence-BERT] Sentence Embeddings using Siamese BERT-Networks (Continue)

- **[DeBERTa] DeBERTa: Decoding-Enhanced BERT with Disentangled-Attention (Complete, [Review](https://qcqced123.github.io/nlp/deberta))**
- [GPT2] Language Models are Unsupervised Multitask Learners (Continue)

- **[ViT] An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale (Complete, [Review](https://qcqced123.github.io/cv/vit))**

- [SwinTransformer] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Continue)

- [CLIP] Learning Transferable Visual Models From Natural Language Supervision (Continue)
 
- [BLIP] BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Continue)


### Recurrent

- **[RNN] Recurrent Neural Network (Complete)**
- **[GRU] Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (Complete)**
- **[LSTM] Long Short-Term Memory (Complete)**
- [ELMO] Deep contextualized word representations (Continue)

### Convolution 

- [ConvNext] A ConvNext for the 2020s (Continue)

- [CoAtNet] Marrying Convolution and Attention for All Data Sizes (Continue)

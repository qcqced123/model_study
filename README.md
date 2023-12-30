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
│		├── mlm.py
│		├── clm.py
│		├── rtd.py
│		├── sbo.py
│	    └── p_tuning.py
│
├── dataset_class
│	├── data_folder  # input your dataset
│	├── dataclass.py
│   └── preprocessing.py
│  
├── model
│	├── abstract_task.py
│   ├── model.py
│   └── model_utils.py
│
├── trainer
│   ├── train_loop.py
│   ├── trainer.py
│   └── trainer_utils.py
│
└── Experiment 
    └──  테스트 수행 과정 및 결과 소개**
```

### Attention
- **[Transformer] Attention Is All You Need (완료, [리뷰](https://qcqced123.github.io/nlp/transformer))**

- [Longformer] Longformer: The Long-Document Transformer (예정)

- [Reformer] Reformer: The Efficient Transformer (예정)

- [ELECTRA] Pre-training Text Encoders as Discriminators Rather Than Generators (예정)

- [SpanBERT] SpanBERT: Improving Pre-training by Representing and Predicting Spans (예정)

- [DistilBERT] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

- [Sentence-BERT] Sentence Embeddings using Siamese BERT-Networks

- **[DeBERTa] DeBERTa: Decoding-Enhanced BERT with Disentangled-Attention (완료, [리뷰](https://qcqced123.github.io/nlp/deberta))**
- [GPT2] Language Models are Unsupervised Multitask Learners (예정)

- **[ViT] An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale (완료, [리뷰](https://qcqced123.github.io/cv/vit))**

- [SwinTransformer] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows(에정)

- [CLIP] Learning Transferable Visual Models From Natural Language Supervision(에정)
 
- [BLIP] BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (에정)


### Recurrent

- **[RNN] Recurrent Neural Network (완료)**
- **[GRU] Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (완료)**
- **[LSTM] Long Short-Term Memory (완료)**
- [ELMO] Deep contextualized word representations (예정)

### Convolution 

- [ConvNext] A ConvNext for the 2020s (에정)

- [CoAtNet] Marrying Convolution and Attention for All Data Sizes (에정)

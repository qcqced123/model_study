# 🔬 ML/DL Experiment Application

### 🖥️ Usage

- **1) select the task type**
- **2) select the model type in config file**
- **3) set your own hyperparameters in config file**
- **4) run the train.py in your terminal**

```bash
# python train.py [Task Type] [Model Name]

python train.py pretrain bert
python train,py pretrain gpt2
python train.py pretrain roformer
python train.py fine_tune superglue
```

### 🖥️ System Info
- CPU: Ryzen 5 5600x, 6 cores 12 threads
- RAM: 32GB
- GPU: RTX 3090 24GB
- OS: Ubuntu 22.04 LTS

### 🖍️ Training Example
Currently, MLM, Distillation Knowledge Task is well perfectly implemented.

RTD and SBO Tasks are implemented, but the pipeline is not optimized. we check out the casue of bottleneck now, and then we will optimize the pipeline. Maybe the cause of bottleneck is the data preprocessing, batching, beacause of our poor system power, especially CPU.

<S> Distillation Knowledge Task is not perfectly implemented. Task is currently experiencing an issue where NaNs are occurring due to training loss(Only CosineEmbeddingLoss, otherwise are normal) after a certain number of forward step. we are currently check out ther reason of this problem. Maybe abnormal text dataset, mixed precision trainiing method are the cause of this problem.</S>

CLM Task is not perfectly implemented. Current version can only train the model by MLE, validate by accuracy. ASAP, we will add other components such as perplexity, BLEU, sliding winodw, etc.

- **1) Masked Language Model (MLM)**  
    https://wandb.ai/qcqced/MaskedLanguageModel?workspace=user-qcqced   

- **2) Causal Language Model (CLM) (not perfectily completed)**  
    https://wandb.ai/qcqced/CasualLanguageModel?workspace=user-qcqced  

- **3) Replaced Token Detection (RTD, need to optimize pipeline)**  
    https://wandb.ai/qcqced/ReplacedTokenDetection?workspace=user-qcqced  
    
- **4) Span Boundary Objective (SBO, need to optimize pipeline)**  
    https://wandb.ai/qcqced/SpanBoundaryObjective?workspace=user-qcqced

- **5) Distillation Knowledge**  
    https://wandb.ai/qcqced/DistillationKnowledge?workspace=user-qcqced

### 📆 Future Developing Schedule

1) Convert current multi-head attention mechanism (for-loop & concatenate) to Parallel Multi-Head Attention Mechanism (Matrix Multiplication with num_heads)

2) Optimize the pipeline of RTD and SBO Task

3) Add Pipeline for famous benchmark Dataset (SFT): SuperGLUE, SQAD, ...  

4) Add other Fine-Tune Task, which is my interest: Sentiment Analysis, QA, Text based Recsys, Text Similarity, Text Generation, etc.  

5) Optimize all the pipeline for numberical stability in mixed precision and max speed in Tensor Core

### 🗂️ Application Structure

![Application Structure](./assets/images/app_structure.png)

This Project implement with Pytorch. The overview structure is as follows.
```plaintext
Experiment Application
│
├── config
│	├── pretrain
│       │    ├──bert.json
│       │    ├──roformer.json  # rotary position embedding
│       │    ├──linear_transformer.json  # linear attention
│       │    ├──distilbert.json
│       │    ├──electra.json
│       │    ├──spanbert.json
│       │    ├──gpt2.json
│       │    ├──deberta.json
│       │    └──deberta_v3.json
│       └── fine_tune
│
├── experiment
│	├── losses
│	│	├── loss.py
│	│	└── distance_metric.py  # for metric learning
│	├── metrics
│	│	└── metric.py
│	├── models
│	│	├── abstract_model.py
│	│	├── attention
│	│	│    ├── bert.py
│	│	│    ├── roformer.py
│	│	│    ├── linear_transformer.py
│	│	│    ├── distilbert.py
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
│	│	└── pooling.py
│	│
│	├── tokenizer
│	│
│	└── tuner
│	    ├── mlm.py  # Masked Language Model (Sub Word Masking/Whole Word Masking) 
│	    ├── clm.py  # Causal Language Model
│	    ├── rtd.py  # Replaced Token Detection
│	    ├── sbo.py  # Span Boundary Objective
│	    ├── lora.py  # Low Rank Adaptation
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

### 📝 Experiment List

#### 🤖 Attention

- **[Transformer] Attention Is All You Need (Complete, [Paper Review](https://qcqced123.github.io/nlp/transformer))**
- **[BERT] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Complete)**
- [Longformer] Longformer: The Long-Document Transformer (Continue)
- [Reformer] Reformer: The Efficient Transformer (Continue)
- **[Roformer] RoFormer: Enhanced Transformer with Rotary Position Embedding (Complete, [Paper Review](https://qcqced123.github.io/nlp/roformer))**  
- [Linear Transformer] Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Continue)
- **[ELECTRA] Pre-training Text Encoders as Discriminators Rather Than Generators (Complete, [Paper Review](https://qcqced123.github.io/nlp/electra))**
- **[SpanBERT] SpanBERT: Improving Pre-training by Representing and Predicting Spans (Complete, [Paper Review](https://qcqced123.github.io/nlp/spanbert))**
- **[DistilBERT] DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (Complete, [Paper Review](https://qcqced123.github.io/nlp/distilbert))**
- [Sentence-BERT] Sentence Embeddings using Siamese BERT-Networks (Continue)
- **[DeBERTa] DeBERTa: Decoding-Enhanced BERT with Disentangled-Attention (Complete, [Paper Review](https://qcqced123.github.io/nlp/deberta))**
- **[DeBERTa-V3] Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing (Complete, [Paper Review](https://qcqced123.github.io/nlp/deberta_v3))**
- **[GPT2] Language Models are Unsupervised Multitask Learners (Complete)**
- **[ViT] An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale (Complete, [Paper Review](https://qcqced123.github.io/cv/vit))**
- [SwinTransformer] Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Continue)
- [CLIP] Learning Transferable Visual Models From Natural Language Supervision (Continue)
- [BLIP] BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (Continue)

#### 🔪 Optimization

- **[LoRA] LoRA: Low-Rank Adaptation of Large Language Models(Complete, [Paper Review](https://qcqced123.github.io/nlp/lora))**

#### 🪢 Recurrent

- **[RNN] Recurrent Neural Network (Complete)**
- **[GRU] Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (Complete)**
- **[LSTM] Long Short-Term Memory (Complete)**
- [ELMO] Deep contextualized word representations (Continue)

#### 🔭 Convolution

- [ConvNext] A ConvNext for the 2020s (Continue)
- [CoAtNet] Marrying Convolution and Attention for All Data Sizes (Continue)
-

#### 📐 Metric Learning

- **[ContrastiveLoss] Dimensionality Reduction by Learning an Invariant Mapping (Complete)**
- **[ArcFace] ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Complete)**

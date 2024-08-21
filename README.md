# LLMs Toolkit: Fine-tuning and RAG Report



## Prerequisition

Before using this project, please make sure you have registered for a Hugging Face API token. The Hugging Face token is required to access certain resources and authenticate with Hugging Face services.
huggingface-cli login

```bash
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

## Installation

```bash
$ pip install -r requirements.txt
```

## Datasets

This repository combined some vietnamese datasets:

1. `alpaca_translate_GPT_35_10_20k.json`: Translate by [VietnamAIHub/Vietnamese_LLMs](https://github.com/VietnamAIHub/Vietnamese_LLMs)

## Finetuned Models

Below is a table of our Finetuned models, detailing the LoRA configurations, and links to access each model.

| Base Model |  LoRA Config | Finetuned Model   |
|------------|-------------|---------------|
|   [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)   |  8-bit precision     | [CallMeMrFern/Llama-7b-vn](https://huggingface.co/CallMeMrFern/Llama-7b-vn) |
| [vietnamese-llama2-7b-40GB](https://huggingface.co/bkai-foundation-models/vietnamese-llama2-7b-40GB)     | 8-bit precision     | [CallMeMrFern/Llama2-7b-40GB_vn](https://huggingface.co/CallMeMrFern/Llama2-7b-40GB_vn) |
|  [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)     |  8-bit precision     | [CallMeMrFern/Llama-2-7b-chat-hf_vn](https://huggingface.co/CallMeMrFern/Llama-2-7b-chat-hf_vn) |

Here are the experiment results using the specified models:

![alt text](images/loss.png)

## 1. Finetune Tenique


1. Single gpus training:

    ```bash
    $ python finetune/lora.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_type llama \
    --data_dir data/general/alpaca_translate_GPT_35_10_20k.json \
    --output_dir finetuned/meta-llama/Llama-2-7b-chat-hf \
    --lora_target_modules '["q_proj", "v_proj"]' \
    --micro_batch_size 1
    ```

2. Distributed training on Multi-GPUs

    ```bash
    $ torchrun --standalone --nnodes=1 --nproc_per_node=2 finetune/lora.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_type llama \
    --data_dir data/general/alpaca_translate_GPT_35_10_20k.json \
    --output_dir finetuned/meta-llama/Llama-2-7b-chat-hf \
    --lora_target_modules '["q_proj", "v_proj"]' \
    --micro_batch_size 1
    ```

## Inference

1.  local log file
    ```bash
    $ python inference/run_exp.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_type llama \
    --data_dir data/general/alpaca_translate_GPT_35_10_20k.json \
    --output_dir finetuned/meta-llama/Llama-2-7b-chat-hf \
    --log_dir ./inference/local.log
    ```
2. gradio interface
    ```bash
    $ python inference/ui.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --model_type llama \
    --output_dir finetuned/meta-llama/Llama-2-7b-chat-hf
    ```


## 2. Retrieval-Augmented Generation (RAG)

The LLMs Toolkit includes a RAG component to enhance the performance of the fine-tuned models. RAG combines the power of pre-trained language models with external knowledge retrieval.

### 2.1 RAG Implementation

Our RAG implementation is inspired by the techniques outlined in the [RAG_Techniques repository](https://github.com/NirDiamant/RAG_Techniques). We've adapted and integrated these techniques to work with our fine-tuned Vietnamese language models.

### 2.2 RAG Architecture

The RAG architecture in our toolkit consists of several key components:

1. **Document Loader**: Handles various file formats (PDF, TXT, DOCX) to ingest external knowledge.
2. **Text Splitter**: Divides large documents into manageable chunks for efficient processing.
3. **Embeddings**: Converts text chunks into vector representations.
4. **Vector Store**: Stores and indexes the embeddings for quick retrieval.
5. **Retriever**: Fetches relevant information based on the input query.
6. **Language Model**: Our fine-tuned LLaMA model that generates responses.
7. **Prompt Template**: Structures the input for the language model.

### 2.3 Knowledge Base

Our knowledge base is built from various Vietnamese language resources, including:

- Academic papers
- News articles
- Wikipedia dumps
- Domain-specific documents

These documents are processed and stored in a vector database for efficient retrieval.

### 2.4 Retrieval Process

The retrieval process follows these steps:

1. User input is converted into an embedding.
2. The vector store is queried to find the most similar documents.
3. Relevant chunks are retrieved and added to the context.
4. The context, along with the original query, is passed to the language model.

### 2.5 RAG Techniques

We've implemented several advanced RAG techniques to improve performance:

1. **Hybrid Search**: Combines keyword-based and semantic search for more accurate retrieval.
2. **Re-ranking**: Applies a secondary ranking to initial search results to improve relevance.
3. **Query Expansion**: Enhances the original query with additional relevant terms.
4. **Contextual Compression**: Reduces retrieved context to the most relevant information.
5. **Multi-Index Retrieval**: Utilizes multiple specialized indexes for diverse knowledge domains.

### 2.6 Integration with Fine-tuned Models

The RAG system is tightly integrated with our fine-tuned LLaMA models:

1. The retriever uses embeddings that are compatible with the Vietnamese language.
2. The prompt template is designed to work effectively with the fine-tuned model's training data.
3. The language model's output is post-processed to ensure coherence with the retrieved information.

### 2.7 Evaluation

We evaluate our RAG system using the following metrics:

- Relevance of retrieved documents
- Answer accuracy
- Response coherence
- Inference time

[Remaining sections unchanged]

## References

1. [VietnamAIHub/Vietnamese_LLMs](https://github.com/VietnamAIHub/Vietnamese_LLMs)
2. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
4. [RAG_Techniques Repository](https://github.com/NirDiamant/RAG_Techniques)

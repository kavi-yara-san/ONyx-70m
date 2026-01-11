

# Onyx

Onyx is a **small, experimental language model** built from scratch using PyTorch.
It is designed for **research, learning, and architectural exploration**, with a focus on
compact Transformers, multihead attention, and Mixture of Experts (MoE).

---

## Features

- Transformer-based language model
- Custom Multi-Head Attention
- Mixture of Experts (MoE) layers
- Custom BPE tokenizer
- Lightweight and hackable codebase
- Suitable for research and experimentation

---

## Project Structure

``` text
main/
├── checkpoint/
├── layers/
│   ├── __init__.py
│   ├── attention.py
│   ├── empeddings.py         
│   ├── MoEmodule.py
│   └── transformer.py
├── tokenizer/
│   ├── merges-o.txt
│   └── vocab-o.json
├── __init__.py
├── architecture.py
├── config.py
├── inference.py
├── .gitignore
├── README.md
├── requirements.txt
└── samp.py

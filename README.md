# CpLoM

This repository provides an end-to-end pipeline for **knowledge graph completion** by combining **semantic generation**, **rule optimization**, **rule validation**, and **filtering**, with the help of large language models (LLMs) like GPT-3.5.

We demonstrate the pipeline on the `family` dataset.

## 🔧 Environment Setup
```python
pip install -r requirements.txt
```

Set your OpenAI API key in ```.env``` file.

---
## 🚀 Pipeline Steps
### 1. Semantic Generation
Generate natural language descriptions for each relation using GPT.
```python
python relation_interpret.py --dataset family
```
### 2. Rule Validator
Filter logically invalid or unsupported rules.
```python
python rule_validator.py --dataset family --rule_path filter_rules
```
### 3. Rule Generation Optimizer
Generate rules using ChatGPT, clean and merge them.
```python
python chat_rule_generator.py --dataset family --model_name gpt-3.5-turbo -f 50 -l 10
python clean_rule.py --dataset family -p gpt-3.5-turbo-top-0-f-50-l-10 --model none
python merge_rules.py --dataset family -p gpt-3.5-turbo-top-0-f-50-l-10
```
### 4. Rule Filter & Ranking
Sort and rank the merged rules.
```python
python sorted_rule.py --dataset family -p gpt-3.5-turbo-top-0-f-50-l-10
python rank_rule.py --dataset family -p gpt-3.5-turbo-top-0-f-50-l-10
```
### Knowledge Graph Completion
Apply filtered rules to infer new facts and complete the KG.
```python
python kg_completion.py --dataset family --input_folder sorted_rules -p gpt-3.5-turbo-top-0-f-50-l-10
```
---
## 📦 Reproduce KGC results with mined rules
```python
python kg_completion.py --dataset family --input_folder ResultRules -p gpt-3.5-turbo-top-0-f-50-l-10
python kg_completion.py --dataset umls --input_folder ResultRules -p gpt-3.5-turbo-top-0-f-50-l-10
python kg_completion.py --dataset wn-18rr --input_folder ResultRules -p gpt-3.5-turbo-top-0-f-50-l-10
python kg_completion.py --dataset yago --input_folder ResultRules -p gpt-3.5-turbo-top-0-f-50-l-10
```

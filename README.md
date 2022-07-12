# IOT-Match

This is the official implementation for the SIGIR 2022 [paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531974)
 "Explainable Legal Case Matching via Inverse Optimal Transport-based Rationale Extraction".


# Overview
As an essential operation of legal retrieval, legal case matching plays a central role in intelligent legal systems. This task has a high demand on the explainability of matching results because of its critical impacts on downstream applications --- the matched legal cases may provide supportive evidence for the judgments of target cases and thus influence the fairness and justice of legal decisions. Focusing on this challenging task, we propose a novel and explainable method, namely IOT-Match, with the help of computational optimal transport, which formulates the legal case matching problem as an inverse optimal transport (IOT) problem. Different from most existing methods, which merely focus on the sentence-level semantic similarity between legal cases, our IOT-Match learns to extract rationales from paired legal cases based on both semantics and legal characteristics of their sentences. The extracted rationales are further applied to generate faithful explanations and conduct matching. Moreover, the proposed IOT-Match is robust to the alignment label insufficiency issue commonly in practical legal case matching tasks, which is suitable for both supervised and semi-supervised learning paradigms. To demonstrate the superiority of our IOT-Match method and construct a benchmark of explainable legal case matching task, we not only extend the well-known Challenge of AI in Law (CAIL) dataset but also build a new Explainable Legal cAse Matching (ELAM) dataset, which contains lots of legal cases with detailed and explainable annotations. Experiments on these two datasets show that our IOT-Match outperforms state-of-the-art methods consistently on matching prediction, rationale extraction, and explanation generation.

# Data
We will provide ELAM to support [CAIL 2022](http://cail.cipsc.org.cn/) explainable legal case matching track. For a fair competition, we will not leak ELAM here. Please stay tuned for CAIL!

For eCAIL, you would like to download it [here](https://drive.google.com/file/d/1ixjnkpGvM8RL7arxFDrCMiVWzJtifQYv/view?usp=sharing).

# Requirements
```python
python == 3.6
```

# Trainining and Evaluation
```python
python xxx.py
```


# Acknowledgement
Please cite the following papers as the references if you use our codes or the processed datasets.

```bib
@inproceedings{10.1145/3477495.3531974,
author = {Yu, Weijie and Sun, Zhongxiang and Xu, Jun and Dong, Zhenhua and Chen, Xu and Xu, Hongteng and Wen, Ji-Rong},
title = {Explainable Legal Case Matching via Inverse Optimal Transport-Based Rationale Extraction},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531974},
doi = {10.1145/3477495.3531974},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {657–668},
numpages = {12},
keywords = {legal retrieval, explainable matching},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```

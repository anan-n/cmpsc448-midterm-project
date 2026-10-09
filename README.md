# Who Wrote It? Identifying LLMs from Their Responses

**Course:** CMPSC 448  
**Author:** Ananya Reddy N

## Project Overview

This project investigates whether machine learning models can identify which large language model (LLM) generated a response. Using a dataset of questions and answers from different LLMs, I trained and evaluated Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to classify responses from six models: ChatGPT, Claude, Gemini, DeepSeek, Llama, and Qwen.

The project also explores how the original question affects classification, whether a classifier generalizes across domains, and whether responses from different LLMs have distinguishable text characteristics.

## Research Questions

- **RQ1:** Can a model identify which LLM generated a response using the response text alone?
- **RQ2:** How does classification performance compare when using the question alone, the response alone, or both?
- **RQ3:** How well does a classifier generalize to a different domain?
- **RQ4:** What statistical differences exist between the response characteristics of different LLMs?

## Dataset

This project uses the TXD-22 Large-Scale Benchmark Dataset, which contains questions, answers, and source labels.

For the main classification experiments, six LLMs were selected, with 2,000 examples per model, resulting in a balanced dataset of 12,000 examples. The selected models are ChatGPT, Claude, Gemini, DeepSeek, Llama, and Qwen.

The data was divided into training, validation, and test sets while grouping normalized questions to reduce overlap between splits.

The original dataset is not included in this repository because of its size. Users should obtain the dataset from its original source and verify its citation and usage terms before reuse.

## Methods

### Neural Network Models

Two neural network architectures were implemented and evaluated:

- **CNN:** Uses convolutional filters with kernel sizes of 3, 4, and 5 to identify patterns in text.
- **LSTM:** Uses a bidirectional Long Short-Term Memory network to learn sequential patterns in text.

The models use word embeddings, dropout, and the Adam optimizer. Experiments were run for five epochs, with model checkpoints selected using validation macro-F1.

### Input Configurations

Three input configurations were compared:

1. **Input only:** Uses the original question.
2. **Output only:** Uses the LLM-generated response.
3. **Input + output:** Uses both the question and the response.

### Additional Experiments

- **RQ3 — Domain generalization:** A TF-IDF and Logistic Regression classifier was trained on non-computer-science examples and evaluated on held-out computer-science examples.
- **RQ4 — Response characteristics:** Response length, lexical diversity, average word length, and sentence count were compared across models using descriptive statistics and Kruskal-Wallis tests.

## Results

### CNN and LSTM Classification

| Input configuration | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Input only | CNN | 17.50% | 0.1610 |
| Input only | LSTM | 17.00% | 0.1446 |
| Output only | CNN | 65.92% | 0.6526 |
| Output only | LSTM | 54.50% | 0.5304 |
| Input + output | CNN | 66.08% | 0.6525 |
| Input + output | LSTM | 43.83% | 0.3895 |

The CNN performed better than the LSTM across all three input configurations in these experiments. Using response text substantially improved classification compared with using questions alone.

The input-plus-output CNN achieved the highest accuracy at 66.08%, while the output-only CNN achieved a marginally higher macro-F1 of 0.6526. The macro-F1 scores were nearly identical, suggesting that adding the original question provided little additional benefit in this experiment.

### RQ3: Domain Generalization

| Evaluation set | Examples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| In-domain non-CS test set | 2,360 | 87.25% | 0.8709 |
| Held-out CS set | 168 | 81.55% | 0.8166 |

Performance decreased on the held-out computer-science examples. However, the held-out set was relatively small, so the results should be interpreted cautiously. This experiment used a separate TF-IDF and Logistic Regression classifier.

### RQ4: Response Characteristics

The descriptive analysis identified differences in response characteristics across the selected models:

- Claude had the highest average response length, at approximately 297 words.
- Llama had the lowest average response length, at approximately 209 words.
- Qwen had the highest average lexical diversity, at approximately 0.703.
- Llama had the lowest average lexical diversity, at approximately 0.532.
- Kruskal-Wallis tests indicated statistically significant differences across models in word count, lexical diversity, average word length, and sentence count (p < 0.001).

These findings indicate that measurable text characteristics vary across models. However, they do not establish which specific characteristics the neural networks used to make their predictions.

## Repository Structure

```text
cmpsc448-midterm-project/
├── README.md
├── prepare_data.py
├── train_models.py
├── extra_credit.py
├── requirements.txt
├── six_llms.csv
├── model_comparison.csv
├── model_comparison_2epochs.csv
├── model_comparison_5epochs.csv
├── label_mapping.json
├── rq3_domain_counts.csv
├── rq3_domain_generalization.csv
├── rq4_feature_summary.csv
├── rq4_statistical_tests.csv
├── report_input_only_CNN.txt
├── report_input_only_LSTM.txt
├── report_output_only_CNN.txt
├── report_output_only_LSTM.txt
├── report_input_plus_output_CNN.txt
└── report_input_plus_output_LSTM.txt
```

The repository contains the source code, curated dataset, model comparison results, and classification reports.

## How to Run

Clone the repository:

```bash
git clone https://github.com/anan-n/cmpsc448-midterm-project.git
cd cmpsc448-midterm-project
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The training and additional experiment scripts expect the curated dataset at `data/six_llms.csv`. If the downloaded repository contains `six_llms.csv` in its root directory, create the expected directory and copy the file:

```bash
mkdir -p data
cp six_llms.csv data/six_llms.csv
```

To reproduce the main classification experiments:

```bash
python train_models.py --epochs 5 --max-per-class 2000
```

The data preparation script can be used to recreate the curated dataset from the original CSV, provided the original dataset is placed at the location expected by the script.

The additional experiments are implemented in `extra_credit.py`. Refer to the script for its required inputs and execution details.

## Limitations

- The experiments cover six selected LLMs and may not generalize to all available models.
- Grouping normalized questions reduces overlap between data splits, but other dataset biases may remain.
- The held-out computer-science evaluation set contains relatively few examples.
- Response characteristics can vary with prompts, datasets, and model versions.
- Classification accuracy alone does not establish reliable LLM authorship detection in real-world settings.

## Conclusion

This project investigates the use of CNNs and LSTMs for identifying the source of LLM-generated responses. The CNN achieved approximately 66% accuracy across six classes, and response text provided substantially more useful classification information than question text alone. Additional experiments identified domain-related performance differences and statistically significant variations in response characteristics.

The findings highlight the potential for identifying patterns associated with different LLMs while also showing the limitations of classification-based authorship identification.

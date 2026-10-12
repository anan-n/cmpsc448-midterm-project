# Who Wrote It? Identifying LLMs from Their Responses

**Course:** CMPSC 448  
**Author:** Ananya Reddy (solo)

## Project Overview

Large Language Models (LLMs) can produce text that differs in linguistic style, response length, vocabulary, and structure. This project investigates whether machine learning models can identify which LLM generated a response.

The project compares Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to classify responses from six LLMs. It also investigates whether the original user prompt helps identify the generating model, whether model identification generalizes to a different domain, and which measurable characteristics distinguish LLM responses.

## Research Questions

This project investigates four research questions:

- **RQ1:** Can we identify which LLM generated a response?
- **RQ2:** Does the user's prompt help identify the LLM?
- **RQ3:** Do LLM fingerprints generalize across tasks or domains?
- **RQ4:** What characteristics distinguish different LLMs?

RQ1 and RQ2 are the main experiments. RQ3 and RQ4 are additional analyses exploring domain generalization and response characteristics.

## Dataset

The project uses the TXD-22 dataset, a publicly available benchmark for identifying AI-generated and mixed-source text.

**Dataset source:** [TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated and Mixed-Source Text Identification](https://data.mendeley.com/datasets/prcjcggtjf/1)

**Citation:**

Iqbal, Md Sadiq, and Mohammod Abul Kashem. (2026). *TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated and Mixed-Source Text Identification*. Mendeley Data, Version 1. DOI: 10.17632/prcjcggtjf.1.

### Selected Data

Six LLMs were selected for the experiments:

- ChatGPT
- Claude
- DeepSeek
- Gemini
- Llama
- Qwen

The curated dataset contains **12,000 examples**, with 2,000 examples per LLM. Each example includes the original input prompt, the model's response, and the corresponding LLM label.

The curated dataset is stored at `data/six_llms.csv` and contains the following columns:

| Column | Description |
|---|---|
| `input` | Original user prompt |
| `output` | LLM-generated response |
| `label` | LLM associated with the response |

The dataset was selected to provide a balanced six-class classification task.

## Methodology

### Data Preparation

The dataset was filtered to retain examples with valid input prompts, outputs, and labels. The six selected LLMs were represented by their respective labels.

For the main classification experiments, the data was split by normalized question groups to reduce the risk of the same prompt appearing in both training and evaluation data.

### Models

Two neural network architectures were implemented using PyTorch:

**Convolutional Neural Network (CNN)**

The CNN uses token embeddings and multiple convolutional filters to learn patterns in text. The convolutional features are combined and used to predict the LLM label.

**Long Short-Term Memory Network (LSTM)**

The LSTM uses token embeddings and a bidirectional recurrent network to capture sequential information from the text. Its learned representation is used for classification.

Both models were trained and evaluated using the same six LLM classes.

The main experiments used a vocabulary limit of 12,000 tokens, a maximum sequence length of 160 tokens, an embedding dimension of 64, and a batch size of 128. The CNN used kernel sizes of 3, 4, and 5, with 64 channels per kernel. The bidirectional LSTM used a hidden size of 64. Both models used dropout of 0.3 and the Adam optimizer with a learning rate of 0.001.

The five-epoch experiments selected checkpoints based on validation macro-F1.

### Evaluation Metrics

The models were evaluated using:

- **Accuracy:** The proportion of correctly classified examples.
- **Macro-F1:** The unweighted average of the F1-scores across all six classes.

Macro-F1 is useful because it measures performance across all classes rather than allowing the largest or easiest class to dominate the evaluation.

## RQ1: Can We Identify Which LLM Generated a Response?

RQ1 evaluates whether the response text alone contains enough information to identify its generating LLM.

The CNN and LSTM models were trained using only the response text.

### Results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CNN | 65.92% | 0.6526 |
| LSTM | 54.50% | 0.5304 |

### Discussion

The CNN outperformed the LSTM on both accuracy and macro-F1. The output-only CNN achieved 65.92% accuracy, demonstrating that the response text contains information that can help distinguish the six LLMs.

However, the classification performance also indicates that identifying the generating model remains challenging. The results suggest that different LLMs leave distinguishable patterns in their responses, although those patterns are not sufficient for consistently accurate classification.

## RQ2: Does the User's Prompt Help Identify the LLM?

RQ2 compares classification performance using three input configurations:

1. **Input only:** The original user prompt.
2. **Output only:** The LLM-generated response.
3. **Input plus output:** The prompt and response combined.

### Results

| Input Configuration | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Input only | CNN | 17.50% | 0.1610 |
| Input only | LSTM | 17.00% | 0.1446 |
| Output only | CNN | 65.92% | 0.6526 |
| Output only | LSTM | 54.50% | 0.5304 |
| Input + output | CNN | 66.08% | 0.6525 |
| Input + output | LSTM | 43.83% | 0.3895 |

### Discussion

The input-only models performed close to the 16.67% accuracy expected from random guessing across six balanced classes. This suggests that the original prompt alone provides limited information about which LLM generated the response.

The output-only CNN performed substantially better than the input-only models. Combining the prompt and response produced the highest CNN accuracy, at 66.08%, but its macro-F1 was essentially identical to that of the output-only CNN.

For the LSTM, combining the input and output reduced performance compared with using the response alone.

Overall, the results indicate that the generated response is more useful than the original prompt for identifying the LLM. Including the prompt did not meaningfully improve CNN macro-F1 and reduced LSTM performance in this experiment.

## RQ3: Do LLM Fingerprints Generalize Across Tasks or Domains?

RQ3 investigates whether a classifier trained on non-computer-science examples can identify LLMs in held-out computer-science examples.

A TF-IDF representation and Logistic Regression classifier were used for this experiment. The dataset was divided by normalized question groups to reduce prompt overlap between training and evaluation data.

### Results

| Evaluation Set | Examples | Unique Questions | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| In-domain non-CS test set | 2,360 | 604 | 87.25% | 0.8709 |
| Held-out computer-science set | 168 | 41 | 81.55% | 0.8166 |

### Discussion

The classifier achieved 87.25% accuracy on the non-computer-science test set and 81.55% accuracy on the held-out computer-science examples. Macro-F1 decreased from 0.8709 to 0.8166.

This decrease suggests that classification performance declined when the classifier was evaluated on the held-out domain. However, the classifier retained relatively strong performance on the computer-science examples, suggesting that some useful model-identification signals may generalize across domains.

There are important limitations. The held-out computer-science set contains only 168 examples and 41 unique questions. Additionally, domain assignments were determined using keyword-based rules rather than manually verified labels. The results should therefore be interpreted as preliminary evidence of domain generalization rather than definitive proof that LLM fingerprints are domain-independent.

## RQ4: What Characteristics Distinguish Different LLMs?

RQ4 examines whether measurable linguistic and structural characteristics differ across LLMs and whether these characteristics can help identify the generating model.

The analysis included six features:

- Word count
- Lexical diversity
- Mean word length
- Sentence count
- Presence of bullet markers
- Presence of code blocks

Descriptive statistics and Kruskal–Wallis tests were used to analyze differences across the six LLMs. A separate Logistic Regression classifier was trained using these handcrafted features.

### Descriptive Results

| LLM | Mean Words | Median Words | Lexical Diversity | Mean Word Length | Mean Sentences |
|---|---:|---:|---:|---:|---:|
| ChatGPT | 246.610 | 244 | 0.626 | 5.875 | 10.772 |
| Claude | 297.336 | 294 | 0.617 | 6.409 | 7.833 |
| DeepSeek | 270.054 | 269 | 0.677 | 6.038 | 13.681 |
| Gemini | 268.226 | 297 | 0.616 | 5.784 | 11.234 |
| Llama | 209.467 | 208 | 0.532 | 5.558 | 7.855 |
| Qwen | 220.831 | 225 | 0.703 | 6.276 | 10.244 |

The descriptive statistics reveal several differences:

- Claude produced the longest responses on average.
- Llama produced the shortest responses and had the lowest lexical diversity.
- Qwen had the highest lexical diversity.
- DeepSeek had the highest average sentence count.
- Claude had the highest average word length.

The bullet-marker and code-block detectors recorded no positive instances for any of the six LLMs in this dataset. These two indicators therefore did not distinguish the models in this experiment.

### Statistical Analysis

Kruskal–Wallis tests were conducted for word count, lexical diversity, mean word length, and sentence count.

| Feature | Kruskal–Wallis H | p-value |
|---|---:|---:|
| Word count | 2976.19 | < 0.001 |
| Lexical diversity | 4555.03 | < 0.001 |
| Mean word length | 2961.35 | < 0.001 |
| Sentence count | 3209.59 | < 0.001 |

All four tests were statistically significant, indicating that the distributions of these characteristics differed across the six LLMs.

These tests establish differences in distributions but do not identify which pairs of LLMs differ significantly. The results should also be interpreted with consideration for the dataset structure and the possibility that multiple LLM responses correspond to the same prompts.

### Feature-Based Classification

A Logistic Regression classifier was trained using the six handcrafted response features.

| Metric | Result |
|---|---:|
| Training examples | 9,604 |
| Test examples | 2,396 |
| Unique test questions | 612 |
| Accuracy | 55.18% |
| Macro-F1 | 0.5422 |

The classifier's performance varied by LLM.

| LLM | Precision | Recall | F1-score |
|---|---:|---:|---:|
| ChatGPT | 0.358 | 0.257 | 0.299 |
| Claude | 0.786 | 0.841 | 0.812 |
| DeepSeek | 0.562 | 0.582 | 0.572 |
| Gemini | 0.339 | 0.307 | 0.322 |
| Llama | 0.747 | 0.840 | 0.790 |
| Qwen | 0.428 | 0.491 | 0.457 |

Claude and Llama were easier to identify using the handcrafted features, while ChatGPT and Gemini were harder to distinguish.

### Discussion

The feature-based classifier achieved 55.18% accuracy and a macro-F1 score of 0.5422. This indicates that simple response characteristics contain useful information for identifying the generating LLM.

However, the feature-based classifier performed below the output-only CNN, which achieved 65.92% accuracy and a macro-F1 score of 0.6526. This suggests that the handcrafted features alone do not capture all the information available in the response text.

The two classifiers were evaluated using different train/test splits, so their scores should not be interpreted as a perfectly controlled comparison. Additionally, this experiment evaluates a classifier based on handcrafted features; it does not directly establish which features the CNN learned or whether removing particular features would reduce CNN performance.

Overall, the results suggest that linguistic characteristics can help distinguish LLMs, but the usefulness of these features varies by model and they do not fully explain classification performance.

## Main Findings

The experiments produced several key findings:

1. **Generated responses contain useful model-identification information.** The output-only CNN achieved 65.92% accuracy and a macro-F1 score of 0.6526.

2. **The CNN outperformed the LSTM in all three input configurations.** This was observed for input-only, output-only, and combined-input classification.

3. **The original prompt provided limited information about the generating LLM.** Input-only accuracy was close to random guessing, and combining input and output did not meaningfully improve CNN macro-F1.

4. **Some model-identification signals generalized to a held-out domain.** The domain-generalization classifier achieved 81.55% accuracy on computer-science examples, although the evaluation set was relatively small.

5. **Response characteristics differed across LLMs.** Statistical tests found significant differences in word count, lexical diversity, mean word length, and sentence count.

6. **Handcrafted features supported model identification but did not match the CNN's performance.** The feature-based classifier achieved 55.18% accuracy and a macro-F1 score of 0.5422.

## Limitations

This project has several limitations:

- The experiments used six selected LLMs, so the results may not generalize to other models.
- Classification performance was imperfect, indicating that the models could not reliably identify every response.
- The domain-generalization evaluation used a relatively small held-out computer-science set.
- The computer-science domain was assigned using keyword-based rules, which may introduce classification errors.
- The feature-based classifier and CNN were evaluated using different train/test splits, limiting the strength of direct performance comparisons.
- The RQ4 analysis evaluated handcrafted features but did not directly remove features from the CNN to measure their individual contribution.
- The descriptive and statistical analyses do not establish that any single feature causes a response to be identifiable.

## Lessons Learned

This project demonstrated that identifying the LLM responsible for a response is a challenging classification problem. Although response text contains useful signals, the performance of different models depends on the architecture and the input representation.

The experiments also showed that adding more input information does not necessarily improve classification. The original prompt provided limited predictive value, and combining it with the response did not meaningfully improve CNN macro-F1.

The domain-generalization experiment highlighted the importance of evaluating classifiers beyond the data distribution used for training. The RQ4 analysis showed that response characteristics can differ significantly across models, but these differences do not fully explain model-identification performance.

Overall, the project provided experience with dataset curation, text preprocessing, CNN and LSTM implementation, model evaluation, domain-generalization experiments, and statistical analysis of language characteristics.

## Project Structure

The repository is organized as follows:

```text
```text
cmpsc448-midterm-project/
├── README.md
├── requirements.txt
├── prepare_data.py
├── train_models.py
├── extra_credit.py
├── rq4_feature_classifier.py
├── six_llms.csv
├── model_comparison_5epochs.csv
├── rq3_domain_generalization.csv
├── rq4_feature_summary.csv
├── rq4_statistical_tests.csv
├── rq4_feature_classifier.csv
├── report_input_only_CNN.txt
├── report_input_only_LSTM.txt
├── report_output_only_CNN.txt
├── report_output_only_LSTM.txt
├── report_input_plus_output_CNN.txt
└── report_input_plus_output_LSTM.txt
```
```

The exact contents of the repository may vary depending on which intermediate results are retained.

## Running the Project

### 1. Install dependencies

Create and activate a Python virtual environment if necessary, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

If the original dataset is available locally, run:

```bash
python prepare_data.py
```

This creates the curated dataset used by the experiments. The original dataset is not included in this repository.

### 3. Train and evaluate the main models

Run:

```bash
python train_models.py --epochs 5 --max-per-class 2000
```

This runs the CNN and LSTM experiments using the three input configurations.

### 4. Run RQ3 and the descriptive/statistical RQ4 analysis

Run:

```bash
python extra_credit.py
```

This runs the domain-generalization experiment and generates the descriptive and statistical response-feature analyses.

### 5. Run the RQ4 feature-based classifier

Run:

```bash
python rq4_feature_classifier.py
```

This trains the Logistic Regression classifier using handcrafted response features and saves its results to `outputs/rq4_feature_classifier.csv`.

## Acknowledgments

The project uses the publicly available TXD-22 dataset for multi-class AI-generated text identification. The dataset authors are credited in the Dataset section.

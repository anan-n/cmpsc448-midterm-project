# cmpsc448-midterm-project
# Who Wrote It? Identifying LLMs from Their Responses

**Course:** CMPSC 448  
**Author:** Ananya Reddy

## Project Overview

This project investigates whether machine learning models can identify which large language model (LLM) generated a response. Using a dataset containing questions and answers from different LLMs, I trained and evaluated Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to classify responses from six models: ChatGPT, Claude, Gemini, DeepSeek, Llama, and Qwen.

The project also examines how much the original question contributes to classification, whether a classifier generalizes across domains, and whether responses from different LLMs have distinguishable text characteristics.

## Research Questions

- **RQ1:** Can a model identify which LLM generated a response using the response text alone?
- **RQ2:** How does classification performance compare when using the question alone, the response alone, or both?
- **RQ3:** How well does a classifier generalize to a different domain?
- **RQ4:** What statistical differences exist between the response characteristics of different LLMs?

## Dataset

The project uses the TXD-22 Large-Scale Benchmark Dataset, which contains questions, answers, and source labels.

For the main classification experiments, I selected six LLMs and created a balanced dataset containing 12,000 examples, with 2,000 examples per model. The data was split into training, validation, and test sets while grouping normalized questions to reduce the chance of the same question appearing across different splits.

The original dataset is not included in this repository because of its size. Please verify the dataset's citation and usage terms before redistributing it.

## Methods

### Text Classification

I implemented and evaluated two neural network architectures:

- **CNN:** Uses convolutional filters with kernel sizes of 3, 4, and 5 to learn patterns in text.
- **LSTM:** Uses a bidirectional LSTM to learn sequential patterns in text.

Both models use word embeddings, dropout, and the Adam optimizer. Models were trained for five epochs, and the best checkpoint was selected using validation macro-F1.

Three input configurations were evaluated:

1. **Input only:** The original question.
2. **Output only:** The LLM-generated response.
3. **Input + output:** The question and response together.

### Additional Experiments

- **RQ3 — Domain generalization:** A TF-IDF and Logistic Regression classifier was trained on non-computer-science examples and evaluated on held-out computer-science examples.
- **RQ4 — Response characteristics:** Response length, lexical diversity, average word length, and sentence count were compared across LLMs using descriptive statistics and Kruskal-Wallis tests.

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

The CNN achieved the strongest overall results. Using the response text was substantially more effective than using the question alone. Combining questions and responses produced the highest accuracy, although its macro-F1 was almost identical to that of the output-only CNN.

These results suggest that response text contains useful patterns for distinguishing LLMs, while the original question alone provides limited classification information.

### RQ3: Domain Generalization

| Evaluation set | Examples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| In-domain non-CS test set | 2,360 | 87.25% | 0.8709 |
| Held-out CS set | 168 | 81.55% | 0.8166 |

Performance decreased on the held-out computer-science examples. However, the held-out set was relatively small, so these findings should be interpreted cautiously. This experiment used a separate TF-IDF and Logistic Regression classifier.

### RQ4: Response Characteristics

The descriptive analysis found differences in response length and lexical characteristics:

- Claude had the highest average response length, at approximately 297 words.
- Llama had the lowest average response length, at approximately 209 words.
- Qwen had the highest average lexical diversity, at approximately 0.703.
- Llama had the lowest average lexical diversity, at approximately 0.532.
- Kruskal-Wallis tests found statistically significant differences across models for word count, lexical diversity, average word length, and sentence count (p < 0.001).

These differences show that response characteristics vary across the selected LLMs, although they do not establish which features the neural network used to make its predictions.

## Repository Structure

```text
cmpsc448_project/
├── data/
│   └── six_llms.csv
├── outputs/
│   ├── model_comparison_5epochs.csv
│   ├── rq3_domain_generalization.csv
│   ├── rq4_feature_summary.csv
│   └── rq4_statistical_tests.csv
├── prepare_data.py
├── train_models.py
├── analyze_features.py
├── rq3_domain_generalization.py
├── requirements.txt
└── README.md
```

The `outputs/` directory also contains model comparison results from other runs and per-model classification reports.

## How to Run

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd cmpsc448_project
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare the curated dataset from the original dataset, after placing the original CSV in the expected location:

   ```bash
   python prepare_data.py
   ```

5. Train and evaluate the models:

   ```bash
   python train_models.py --epochs 5 --max-per-class 2000
   ```

The additional RQ3 and RQ4 scripts can be run separately. Refer to the scripts for their specific arguments and input requirements.

## Limitations

- The experiments cover six selected LLMs and may not generalize to every available model.
- The dataset contains repeated questions, and grouping by normalized question helps reduce, but may not eliminate, potential dataset biases.
- Domain-generalization results are based on a relatively small held-out computer-science set.
- Text characteristics can differ across datasets, prompts, and model versions.
- Classification performance alone does not establish that a model can reliably identify LLM authorship in real-world settings.

## Conclusion

This project demonstrates that CNNs can identify the source LLM from response text with approximately 66% accuracy across six classes. Question-only classification performed close to chance, while including the response text substantially improved performance. Additional experiments revealed domain-related performance differences and statistically significant variation in response characteristics across the selected LLMs.

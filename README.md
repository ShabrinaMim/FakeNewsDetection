# Fake News Detection (Machine Learning + NLP)

## Overview
The rapid spread of misinformation has made automated fake news detection an important real-world problem. This project applies machine learning and Natural Language Processing (NLP) techniques to classify news articles as either **real** or **fake**. 

The goal is to analyze how different preprocessing strategies and feature extraction techniques impact model performance, and to compare a wide range of ML algorithms on the same dataset.

The workflow includes:
- Text cleaning and normalization
- NLP-based feature engineering
- Model training and evaluation
- Comparison across multiple classifiers

---

## Dataset
The dataset contains approximately **20,800 news articles**, each labeled as either *reliable (0)* or *unreliable (1)*.

Dataset link:
https://www.kaggle.com/c/fake-news/data

**Attributes:**
- `id` – unique identifier
- `title` – news headline
- `author` – article author
- `text` – main content
- `label` – 0 (True) or 1 (Fake)

---

## Preprocessing Strategy

Two different approaches were used to evaluate the effect of NLP preprocessing:

### **Approach 1 – Basic Cleaning**
- Used only the `text` column
- Removed rows with missing content
- Applied TF-IDF vectorization directly

### **Approach 2 – Enhanced NLP Pipeline**
- Combined `title`, `author`, and `text` into a single field
- Filled missing values with blank strings
- Removed stop-words and special characters
- Applied tokenization and lemmatization
- Performed CountVectorizer + TF-IDF transformation

**Observation:**
Advanced preprocessing significantly improves performance, especially for models sensitive to noisy text.

---

## Models Evaluated
The following machine learning algorithms were trained and compared:

1. Passive Aggressive Classifier
2. Logistic Regression
3. Support Vector Machine (Linear Kernel)
4. Gradient Boosting Classifier 
5. Random Forest 
6. Decision Tree 
7. AdaBoost 
8. XGBoost 
9. Multinomial Naive Bayes 
10. K-Nearest Neighbors 
11. Multi-Layer Perceptron (MLP)

This range includes:
- Linear models 
- Tree-based models 
- Boosting methods 
- Neural networks 
- Probabilistic classifiers 

---

## Results Summary

### ✔ Key Findings
- **10 out of 11 models** improved significantly when using the enhanced preprocessing pipeline. 
- Better text cleaning **reduced false negatives** for most models. 
- Models like **Passive Aggressive**, **Logistic Regression**, **SVM**, and **Gradient Boosting** consistently provided the best results.

### ✔ Best-Performing Models (Approximate Metrics)
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Passive Aggressive | ~98% | ~98% | ~98% | ~98% |
| Logistic Regression | ~98% | ~98% | ~98% | ~98% |
| SVM (Linear) | ~98% | ~98% | ~98% | ~98% |
| Gradient Boosting | ~97% | ~97% | ~97% | ~97% |

### ✔ Important Observation  
A good fake news classifier must keep **false negatives low**, because classifying fake news as real can have harmful consequences. The best models achieved both:
- **High accuracy**
- **Low false negatives**

### ✖ Underperforming Models
- **KNN** showed low accuracy and large false-negative counts. 
- **Multinomial Naive Bayes** produced many false negatives despite moderate accuracy. 

---

## How to Run the Project

1. Clone this repository:
    ```bash
    git clone https://github.com/ShabrinaMim/FakeNewsDetection.git
    ```
2. Download `train.csv` and `test.csv` from Kaggle. 
3. Create a folder called **fake-news** inside the project directory.
4. Place the downloaded CSV files inside the `fake-news` folder.
5. Open the Jupyter notebook and run all cells:
    ```bash
    jupyter notebook
    ```

---

## Technologies Used
- Python 
- Scikit-Learn 
- Pandas 
- NumPy 
- Matplotlib/Seaborn 
- Natural Language Processing (NLP) 
- TF-IDF and CountVectorizer

---

## Author
**Shabrina Sharmin Mim**
GitHub: https://github.com/ShabrinaMim
Email: mimba007@gmail.com



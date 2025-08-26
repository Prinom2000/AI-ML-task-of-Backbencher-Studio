
# 📌 IMDB Movie Review Sentiment Analysis
Main Dataset Source: https://www.kaggle.com/datasets/mantri7/imdb-movie-reviews-dataset
And here I include a part of cleaned dataset. Becouse the main cleaned dataset is more then 25 MB which is not supported by github.


## 🔹 Project Overview
This project performs **sentiment classification** on the IMDB movie reviews dataset — predicting whether a review is **Positive** or **Negative**.  
Two approaches have been implemented:
1. **Machine Learning Models** – Logistic Regression, Naive Bayes (TF-IDF features)
2. **Deep Learning Model** – Bidirectional LSTM (Word Embeddings)

---

## 🔹 Approach
1. **Data Preparation**
   - Load dataset from Kaggle
   - Remove HTML tags, punctuation, and numbers
   - Convert text to lowercase
   - Remove stopwords (NLTK)
   - Train-test split (80/20)

2. **Feature Extraction**
   - **Machine Learning** → TF-IDF (max_features=5000)
   - **Deep Learning** → Tokenizer + Embedding Layer

3. **Model Training**
   - Logistic Regression (Hyperparameter tuning with GridSearchCV)
   - Multinomial Naive Bayes
   - Bidirectional LSTM (Keras, EarlyStopping)

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix visualization
   - Model saving:
     - Logistic Regression → `.pkl`
     - LSTM → `.h5` + tokenizer `.pkl`

5. **Demo Script**
   - Accepts user input and predicts sentiment.

---

## 🔹 Tools & Libraries
- **Data Handling**: Pandas, NumPy
- **Text Processing**: NLTK, re
- **Machine Learning**: Scikit-learn
- **Deep Learning**: TensorFlow / Keras
- **Visualization**: Matplotlib, Seaborn
- **Model Saving**: joblib, pickle

---

## 🔹 Results
| Model                     | Accuracy | Precision | Recall | F1-score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression (TF-IDF) | ~88%     | 0.88      | 0.88   | 0.88     |
| Multinomial Naive Bayes   | ~86%     | 0.86      | 0.86   | 0.86     |
| Bidirectional LSTM        | ~90%     | 0.90      | 0.90   | 0.90     |


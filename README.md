Hotel Reviews Sentiment Analysis (NLP)
This project implements a "Natural Language Processing (NLP) based Sentiment Analysis system" for hotel reviews using "Machine Learning".  
It includes "model training, evaluation, artifact saving" and a "Streamlit web application" for real-time sentiment prediction.
Author
Ferdinand C. Sungoh,Batch-11  
Project Objective
To classify hotel reviews into:
-Positive.
-Negative.
using "TF-IDF features" and "classic machine learning models" and deploy the best model through an interactive Streamlit application.
Project Structure
HOTELREVIEW/
│
├─ Hotel_Reviews.CSV 
├─ app_streamlit.py 
├─ train_model.ipynb/train.py
│
└─ sentiment_artifacts/
├─ hotel_sentiment_pipeline.joblib 
├─ model_comparison.csv
├─ eda_summary.txt
└─ best_confusion_matrix.png 

How It Works (Pipeline Overview)
1) Dataset Columns
The script auto-detects these columns (common Kaggle schema):
-Positive Review**: `Positive_Review`
-Negative Review**: `Negative_Review`
-Rating (optional)**: `Reviewer_Score`
2)Exploratory Data Analysis (EDA)
The code prints and visualizes:
-Dataset shape and data types
-Missing value counts (top 20)
-Rating distribution (if rating column exists)
-Review length statistics (positive & negative)
-Histograms for review length
It also saves a summary to:
 `sentiment_artifacts/eda_summary.txt`
3) Label Creation (Feature Engineering)
The dataset contains separate 'Positive' and 'Negative' review fields per hotel review entry.
This script converts them into a supervised dataset:
-Each non-empty 'Positive_Review' becomes one training sample with label '1 (Positive)'
-Each non-empty 'Negative_Review' becomes one training sample with label '0 (Negative)'
-The placeholder texts `"No Positive"` and `"No Negative"` are excluded
Result:
-`X_text`=array of review texts  
-`y`=labels (0 = Negative, 1 = Positive)
4)Text Vectorization (TF-IDF FeatureUnion)
The model uses two feature types combined together:
1.Word TF-IDF**
   - unigrams+bigrams`(1,2)`
   -`stop_words="english"`
   -limited to `max_features=60000`
2. Character TF-IDF
   -`char_wb` analyzer
   -ngrams `(3,5)`
   - helpful for misspellings and short phrases
These are combined via `FeatureUnion`.
5) Models Trained
Three models are trained and compared:
-Logistic Regression(balanced class weights)
-LinearSVC(balanced class weights)
- Multinomial Naive Bayes
A train/test split is used:
-80% train, 20% test
-`stratify=y` for class balance
6)Evaluation + Overfitting Check
For each model  the script computes:
- Accuracy
- Precision
- Recall
- F1-score
It also calculates an overfitting indicator:
Overfit Gap=Train F1 − Test F1**
The best model is chosen using:
-Highest Test F1-score
Outputs saved:
-`sentiment_artifacts/model_comparison.csv`
-`sentiment_artifacts/best_confusion_matrix.png`
7) Saving the Best Model (Reusable Artifact)
The best pipeline is saved using Joblib to:
- `sentiment_artifacts/hotel_sentiment_pipeline.joblib`
Saved artifact contains:
-trained pipeline
-label map
-best model name
NOTE:Installation
Create an environment and install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn joblib
______________________________________________________________________

app_streamlit.py

1.Hotel Reviews Sentiment Analysis (NLP)
This project implements a "Natural Language Processing (NLP) based Sentiment Analysis system" for hotel reviews using "Machine Learning".  
It includes "model training,evaluation,artifact saving" and a "Streamlit web application" for real-time sentiment prediction.
2.Author
Ferdinand C. Sungoh,Batch-11  
3.Project Objective
To classify hotel reviews into:
-Positive.
-Negative.
using "TF-IDF features" and "classic machine learning models" and deploy the best model through an interactive Streamlit application.
4.Project Structure

HOTELREVIEW/
│
├─ Hotel_Reviews.CSV 
├─ app_streamlit.py 
├─ train_model.ipynb/train.py
│
└─ sentiment_artifacts/
├─ hotel_sentiment_pipeline.joblib 
├─ model_comparison.csv
├─ eda_summary.txt
└─ best_confusion_matrix.png 

5.How the Model is Trained
5.a. Data Preparation
- Uses "Positive_Review" and "Negative_Review" columns.
- Each review becomes a training sample:
  - Positive_Review->"Positive (1)"
  - Negative_Review->"Negative (0)"
- Ignores placeholder texts like "No Positive" or "No Negative".

6.Text Feature Engineering
6.a.Uses a "FeatureUnion" of:
- "Word-level TF-IDF"
  - Unigrams & Bigrams (1,2)
- "Character-level TF-IDF"
  - Character n-grams (3–5)
This improves robustness to spelling variations and short reviews.
7.Models Trained
The following models are trained and compared:
- Logistic Regression
- Linear Support Vector Machine (LinearSVC)
- Multinomial Naive Bayes
8.Evaluation Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Overfitting gap (Train F1 − Test F1)
9.Best model is selected based on highest Test F1-Score**
10.Saved Artifacts
After training, the following are saved:
- `hotel_sentiment_pipeline.joblib`->trained pipeline
- `model_comparison.csv`->performance comparison
- `best_confusion_matrix.png`
- `eda_summary.txt`
11.Streamlit Web Application
The Streamlit app allows users to:
Prediction
- "Single Review Prediction"
- "Bulk Prediction"(one review per line)
- Download predictions as CSV
12.Reports
- Best model used
- Model comparison table
- EDA summary
- Confusion matrix

Reference:

https://www.kaggle.com/code/jonathanoheix/sentiment-analysis-with-hotel-reviews/notebook#Conclusion

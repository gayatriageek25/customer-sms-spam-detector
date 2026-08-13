# SMS Spam Detection Using Machine Learning

A simple NLP and machine-learning project that classifies an SMS message as **Spam** or **Not Spam (Ham)**.

## Problem
Unwanted SMS messages can contain promotions, scams, phishing attempts, or misleading offers. This project demonstrates how NLP and supervised machine learning can automatically classify messages.

## Approach
1. Load the UCI SMS Spam Collection dataset.
2. Clean and preprocess message text.
3. Convert text into TF-IDF numerical features.
4. Train a Logistic Regression classifier.
5. Evaluate the model using accuracy, precision, recall and F1-score.
6. Use the trained model to classify new messages.
7. Run an interactive Streamlit interface.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NLP / TF-IDF
- Streamlit
- Joblib
- GitHub

## Dataset
This project uses the **SMS Spam Collection** from the UCI Machine Learning Repository.
Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset contains 5,574 labelled SMS messages.

## How to Run

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Train:
```bash
python train_model.py
```

Run the app:
```bash
streamlit run app.py
```

## Example
Input: "Congratulations! You have won a free prize. Click the link to claim now."

Output: **Spam**

Input: "Are you coming to class today?"

Output: **Not Spam**

## Project Status
Completed as a beginner/intermediate machine-learning NLP project.

## Author
Add your name here.

# **Cyberbullying Detection on Social Media**

Hi there! Welcome to my project on **cyberbullying detection**. This project is close to my heart because of its real-world relevance—cyberbullying is a huge issue today, and I wanted to explore how machine learning could help tackle it. Below, I’ll walk you through what I’ve done, the tools I used, and the insights I uncovered.

---

## **Why This Project?**
Social media can be a double-edged sword. While it connects people, it also gives rise to harmful behaviors like cyberbullying. I wanted to analyze this problem from a data-driven perspective and see if I could classify tweets as **cyberbullying** or **non-cyberbullying** using sentiment analysis and machine learning.

---

## **What’s in the Dataset?**
The dataset I worked with is from MNIST and contains 20,000 tweets, each labeled as **cyberbullying (1)** or **non-cyberbullying (0)**. It includes:
- **Raw tweet text**
- **Annotation labels** (binary)
- Features like **negative, neutral, and positive sentiment scores**, calculated using VADER Sentiment Analysis.

This gave me a strong starting point for building models and interpreting results.

---

## **How I Approached the Problem**
Here’s a quick breakdown of my workflow:

### **1. Data Preprocessing**
- Cleaned the tweet text by removing unwanted characters, converting everything to lowercase, and stripping out extra spaces.
- Added **sentiment scores** (negative, neutral, positive, compound) using VADER Sentiment Analysis.
- Labeled tweets: 
  - **Cyberbullying (1)**: Compound score < 0
  - **Non-cyberbullying (0)**: Compound score >= 0

### **2. Building Machine Learning Models**
I experimented with four models:
1. **SGDClassifier** (for speed and simplicity)
2. **Logistic Regression**
3. **Random Forest**
4. **Gradient Boosting**

To evaluate each model, I used **Stratified K-Fold Cross-Validation**. This ensured the dataset's class imbalance didn’t skew my results.

### **3. Evaluation Metrics**
I didn’t just stop at accuracy—cyberbullying detection is a serious topic, so I looked at:
- **Precision**: How many predicted cyberbullying cases were correct?
- **Recall**: How many actual cyberbullying cases were caught by the model?
- **F1-Score**: The balance between precision and recall.

---

## **Results That Got Me Excited**
Here’s how the models performed:

| Model                       | Accuracy| Precision | Recall |F1-Score |
|--------------------         |---------|---------- |--------|---------|
| **SGDClassifier**           | 96.29%  | 97.71%    | 95.06% | 96.51%  |
| **Logistic Regression**     | 95.65%  | 97.30%    | 94.20% | 95.72%  |
| **Random Forest**           | 98.87%  | 98.92%    | 98.89% | 98.90%  |
| **Gradient Boosting**       | 98.35%  | 98.45%    | 98.37% | 98.41%  |

**Takeaway**: The Random Forest model knocked it out of the park! It achieved an F1-Score of **98.90%**, making it my go-to choice.

---

## **Visual Insights**
All the visualizations have been combined into a single image to make it easier to navigate and analyze. The combined visualizations include:
- **Accuracy Comparison**
- **Precision Comparison**
- **Recall Comparison**
- **F1 Score Comparison**
- **Confusion Matrix**
- **Class Distribution**

[**Click here to view the combined visualizations**](https://github.com/SrinidhV/Cyberbullying-Detection-on-Social-Media-A-Comparative-Analysis-of-Machine-Learning-Models/blob/main/Entire_viz.png)

---

## **What I Learned**
- **Sentiment Analysis**: VADER is a powerful tool for sentiment scoring, and the `compound` score proved crucial for labeling.
- **Model Comparison**: Not all machine learning models are created equal. Random Forest and Gradient Boosting outshone simpler linear models like SGD.
- **Visualization**: Communicating findings visually helped me understand the data and model performance better.

---

## **Future Plans**
This project isn’t over! Here’s what I’m planning next:
1. **Feature Engineering**: Add more features like hashtags, user mentions, and tweet length.
2. **Real-Time Deployment**: Build a REST API for real-time classification of tweets.
3. **Deep Learning**: Experiment with LSTM or transformers for even better accuracy.

---

# RNN Sentiment Analysis of Metacritic's Best Movies and Reviews - 2025

## Project Overview
This project explores sentiment analysis using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers to classify movie reviews. The goal is to analyze audience sentiment, enabling businesses to derive insights for better decision-making in marketing, content curation, and customer engagement.

## Authors
- **Aayush Garg** (055001)
- **Priyanka Goyal** (055034)

## Technology Stack
- **Programming Language:** Python
- **Frameworks & Libraries:** TensorFlow, Keras, NumPy, Pandas, Seaborn, Matplotlib, Streamlit
- **Model Type:** Recurrent Neural Network (RNN) with LSTM

## Data Source
- **Dataset:** Metacritic Movies and Reviews
- **Movies:** 16K+
- **Reviews:** 667K+ (Critic & User)
- **Includes:** Ratings, genres, directors, cast, sentiment scores

## Project Objective
- Build an RNN-based sentiment analysis model that classifies movie reviews as positive or negative.
- Help businesses analyze audience sentiment to refine content strategies, improve marketing campaigns, and enhance customer engagement.

## Problem Statement
- Understanding customer sentiment is vital for tracking opinions and market trends.
- Traditional NLP models struggle with long-term dependencies, affecting accuracy.
- Basic RNNs face vanishing gradient problems, making them ineffective for long sequences.
- LSTM networks overcome these issues by preserving sequential dependencies, leading to better sentiment classification.

## Business Applications
- **Enhanced Content & Service Positioning:** Understanding customer emotions to refine product strategies.
- **Optimized Marketing Campaigns:** Targeted advertisements based on sentiment trends.
- **Brand Reputation Management:** Identifying negative feedback for proactive resolution.
- **Personalized Recommendations:** Improving user engagement on streaming platforms.

## Data Preprocessing
- **Cleaning:** Removed null values, special characters, and extra spaces.
- **Text Processing:** Tokenization, stopword removal, lemmatization.
- **Encoding:** Converted sentiment labels to numeric, tokenized text.
- **Splitting:** 80-20% train-test split.

## Model Selection & Architecture
- **Embedding Layer:** Converts words into dense vectors.
- **LSTM Layers (64 & 32 units):** Learn sequential patterns.
- **Dropout (0.5):** Prevents overfitting.
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Evaluation Metric:** Mean Absolute Error (MAE)

## Model Performance
- **Test MAE:** 0.7515
- **Key Challenges:**
  - Class Imbalance: Neutral reviews dominate the dataset, affecting predictions.
  - Overfitting: Training loss is lower than validation loss.
  - Difficulty with Extreme Sentiments: Struggles with highly positive/negative reviews.

## Business Insights & Recommendations
- **Refine Sentiment Detection:** Improve handling of sarcasm, mixed reviews, and emotional intensities.
- **Address Class Imbalance:** Implement oversampling (SMOTE) or weighted loss functions.
- **Improve Data Representation:** Use BERT or Word2Vec for better contextual understanding.
- **Optimize Targeted Marketing:** Use sentiment insights to adjust branding strategies.
- **Enhance Brand Reputation Management:** Identify negative trends early for proactive resolution.
- **Align User & Critic Perceptions:** Identify discrepancies to refine marketing efforts.

## Future Enhancements
- **Advanced Sentiment Representation:** Implement contextual embeddings (BERT, ELMo) for nuanced sentiment detection.
- **Fine-Tuning for Business Applications:** Hyperparameter optimization for better accuracy.
- **Real-Time Sentiment Dashboard:** Implement Streamlit-based visualization for real-time analysis.
- **Integration with Business Intelligence Tools:** Enhance decision-making with sentiment-driven insights.


## License
This project is licensed under the APACHE License.

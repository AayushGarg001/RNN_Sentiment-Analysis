## 🎬 RNN Sentiment Analysis of Metacritic's Best Movies and Reviews - 2025

### 📌 Project Overview
This project leverages Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) layers to perform sentiment analysis on movie reviews. The aim is to analyze audience sentiment, helping businesses make data-driven decisions in marketing, content curation, and customer engagement.

### 👥 Authors
- **Aayush Garg** (055001)
- **Priyanka Goyal** (055034)

### 🛠️ Technology Stack
- **Programming Language:** Python 🐍
- **Frameworks & Libraries:** TensorFlow, Keras, NumPy, Pandas, Seaborn, Matplotlib, Streamlit
- **Model Type:** Recurrent Neural Network (RNN) with LSTM

### 🎞️ Data Source
- **Dataset:** Metacritic Movies and Reviews
- **Movies:** 16K+
- **Reviews:** 667K+ (Critic & User)
- **Includes:** Ratings, genres, directors, cast, sentiment scores

### 🎯 Project Objectives
✅ Develop an RNN-based model to classify movie reviews as positive or negative.  
✅ Provide actionable insights to businesses for content refinement, marketing improvement, and customer engagement.  

### 🚀 Why This Matters?
- Understanding customer sentiment is essential for tracking opinions and trends.
- Traditional NLP models struggle with long-term dependencies, leading to inaccurate predictions.
- Basic RNNs suffer from vanishing gradient problems, making them ineffective for long sequences.
- LSTM networks solve this by preserving sequential dependencies, resulting in superior sentiment classification.

### 📊 Business Applications
- 🎭 **Enhanced Content & Service Positioning:** Align content with audience emotions.
- 📢 **Optimized Marketing Campaigns:** Run targeted ad campaigns based on sentiment trends.
- 📉 **Brand Reputation Management:** Detect and respond to negative reviews early.
- 🎞 **Personalized Recommendations:** Enhance user engagement on streaming platforms.

### 🔍 Data Preprocessing Pipeline
✔ **Cleaning:** Removed null values, special characters, extra spaces.  
✔ **Text Processing:** Tokenization, stopword removal, lemmatization.  
✔ **Encoding:** Converted sentiment labels to numeric, tokenized text.  
✔ **Splitting:** 80-20% train-test split.  

### 🏗️ Model Architecture
📌 **Embedding Layer:** Converts words into dense vectors.  
📌 **LSTM Layers:** (64 & 32 units) Learn sequential patterns.  
📌 **Dropout (0.5):** Prevents overfitting.  
📌 **Loss Function:** Mean Squared Error (MSE)  
📌 **Optimizer:** Adam  
📌 **Evaluation Metric:** Mean Absolute Error (MAE)  

### 🔬 Model Performance
📊 **Test MAE:** 0.7515  

### ⚠️ Challenges Faced
⚡ **Class Imbalance:** Neutral reviews dominate, affecting prediction accuracy.  
⚡ **Overfitting:** Training loss < Validation loss.  
⚡ **Difficulty with Extreme Sentiments:** Struggles with highly positive/negative reviews.  

### 📈 Business Insights & Recommendations
🔹 **Refine Sentiment Detection:** Improve handling of sarcasm and mixed emotions.  
🔹 **Balance the Dataset:** Implement SMOTE (Synthetic Minority Over-sampling Technique).  
🔹 **Better Word Representation:** Use BERT or Word2Vec for improved context understanding.  
🔹 **Targeted Marketing Strategies:** Utilize sentiment trends for more effective campaigns.  
🔹 **Proactive Brand Management:** Detect and respond to negative feedback faster.  
🔹 **Understand Audience Perception:** Align critic and user opinions for marketing optimization.  

### 🔮 Future Enhancements
✨ **Advanced Sentiment Representation:** Integrate BERT, ELMo for better contextual sentiment detection.  
✨ **Hyperparameter Tuning:** Optimize model performance.  
✨ **Real-Time Sentiment Dashboard:** Develop Streamlit-based visualization for real-time insights.  
✨ **BI Tool Integration:** Combine sentiment analysis with business intelligence tools for strategic decision-making.  

### 📜 License
This project is licensed under the Apache License.  


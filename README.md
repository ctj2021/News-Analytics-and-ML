# News-Analytics-and-Machine-Learning
Quantitative Equity Trading Signal Based on Financial News Headlines Sentiment Classification
Apply NLP techniques to building financial sentiment classification with various fine-tuned supervised machine learning models, then utilize the predicted signals in quant equity trading.

Machine learning (ML) and Natural Language Processing (NLP) have been developing for wide use in many areas. In this project, we applied and compared various ML and NLP techniques through building sentiment classification models on recent financial news headlines data from Kaggle. 

We referred to huggingface (2022) transformers pipeline and Sentence Transformer pretrained model (Reimers, 2022) to conduct NLP like text preprocessing, sentiment analysis, and word embedding using both TF-IDF (Term Frequency-Inverse Document Frequency) and BERT (Bidirectional Encoder Representations from Transformers). 

Apart from above, we also fine-tuned (Tran, 2020) different ML models, including Naive Bayes and Random Forest, to see how we can efficiently classify the sentiment of financial big data, and transform such sentimental results into quantitative trading signals.

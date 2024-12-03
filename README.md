# OmniCommerce AI 
About My Project
Project Overview
This project is designed to provide an integrated solution for translating text, recommending products, and processing images in a user-friendly interface. It aims to enhance user interaction through predictive analytics and personalized recommendations while ensuring seamless handling of textual and visual data.

Problem Statement
The goal of this project is to predict whether a user visiting an E-commerce platform will convert into a customer or not. The dataset provided contains information about users, and the target variable is whether the user has converted or not (has_converted). Additionally, a recommendation system will be built to suggest products to users based on their preferences.
Key Components
1.	Translation Module: This component allows users to input text in one language and receive a translation in another. It supports multiple languages and leverages a machine learning-based translator to ensure accurate and efficient translations.
2.	Recommendation System: The recommendation system analyzes user data to suggest relevant products. By employing collaborative filtering techniques and machine learning algorithms, it predicts user preferences based on historical interactions, enhancing the shopping experience.
3.	Image Preprocessing: This module focuses on preparing images for analysis and recommendation purposes. It includes techniques like resizing, normalization, and data augmentation to ensure high-quality images that improve the accuracy of predictions.
4.	Text Preprocessing: Text preprocessing ensures that input text is clean and structured before it is used in translation or recommendation tasks. This includes tokenization, stop-word removal, and normalization, which enhance the performance of natural language processing tasks.

   
Technical Implementation
•	Libraries Used: The project utilizes popular libraries such as Streamlit for creating the web interface, Pandas for data manipulation, and scikit-learn for machine learning algorithms. Additionally, the project leverages the Google Translate API for translation tasks.

## Libraries Used
- Streamlit
- pandas
- numpy
- plotly
- matplotlib
- seaborn
- googletrans
- sentence_transformers
- scikit-learn
- scipy
- easyocr
- nltk
- spacy
- wordcloud

•	Data Sources: The recommendation system relies on historical user data, which includes purchase history and user interactions with products. Image datasets are used for image processing tasks.
DataSet: https://drive.google.com/drive/folders/1ATULlRKrSensZHs2SxaT7y0b68Rc1vQA


## Features

Highlight key features of your project, including product prediction, visualizations, and any other functionalities.

## Installation

Provide instructions on how to install and set up the project dependencies.
Version python 10
```bash
pip install -r requirements.txt.
```
Usage
streamlit run Final.py

**Machine Learning Models**

RandomForestClassifier
KNeighborsClassifier
LogisticRegression
Support Vector Machine (SVM)
Include information on training, evaluation metrics, and model performance.

**Natural Language Processing**
Text preprocessing
Sentiment analysis
Translation

**Computer Vision**
Image processing with OpenCV
Optical Character Recognition (OCR) using easyocr
Describe the machine learning models used for product prediction.



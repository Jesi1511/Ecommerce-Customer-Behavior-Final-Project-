import streamlit as st
import re
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import datetime
from googletrans import Translator, LANGUAGES
#from googletrans import Translator
#import googletrans 
#from translate import Translator
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import csv
from sklearn.linear_model import LogisticRegression
from PIL import Image,ImageFilter,ImageEnhance,ImageOps
import easyocr
import cv2
import nltk
from nltk.data import find
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from wordcloud import WordCloud
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

st.set_page_config(layout='wide',page_title="Ecommerce Customer Behavior Final_Project By Jesima Parvin")


def back_img(image):
    with open(image, "rb") as image_file:
        encode_str = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encode_str.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

back_img("pg.jpg") 

def color_patt(sline_in,i_in):
    wch_colour_box = (0, 204, 102)
    fontsize = 25
    sline = sline_in #"text"
    lnk = '<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.12.1/css/all.css" crossorigin="anonymous">'
    i = i_in 
    htmlstr = f"""<p style='background-color: rgba({wch_colour_box[1]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[1]}, 0.75); 
                                font-size: {fontsize}px; 
                                border-radius: 10px; 
                                padding-left: 12px; 
                                padding-top: 18px; 
                                padding-bottom: 18px; 
                                line-height: {fontsize * 1.5}px;'> <!-- Adjusted line height -->
                                <span style='color: black; font-size: {fontsize+5}px;'>{i}</span><br>
                                <span style='color: black; font-size: {fontsize}px; margin-top: 0;'>{sline}</span></p>"""

    st.markdown(lnk + htmlstr, unsafe_allow_html=True)




def load_data():
    try:
        # Use more robust CSV loading parameters
        amazon_data = pd.read_csv(
            r"C:\Users\skath\OneDrive\Desktop\DTM9\Ecommerce-Customer-Behavior-Final-Project--main\recommend.csv",
            encoding='ISO-8859-1',
            on_bad_lines='warn',  # Warn about bad lines instead of skipping
            quoting=csv.QUOTE_MINIMAL,  # Change quoting to QUOTE_MINIMAL
            low_memory=False  # This can help with large files
        )
        print("Columns in the loaded data:", amazon_data.columns)
        amazon_data.drop_duplicates(inplace=True)
        amazon_data.columns = amazon_data.columns.str.strip()
        le = LabelEncoder()
        amazon_data["product_name_label"] = le.fit_transform(amazon_data["product_name"])
        return amazon_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def recommend_products(df, user_id_encoded):
    tfidf = TfidfVectorizer(stop_words='english')
    df['about_product'] = df['about_product'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['about_product'])

    user_history = df[df['user_id_encoded'] == user_id_encoded]

    indices = user_history.index.tolist()

    if indices:
        cosine_sim_user = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)
        products = df.iloc[indices]['product_name']
        indices = pd.Series(products.index, index=products)

        similarity_scores = [(i, score) for (i, score) in enumerate(cosine_sim_user[-1]) if i not in indices]
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        top_products = [i[0] for i in similarity_scores[1:6]]
        recommended_products = df.iloc[top_products]['product_name'].tolist()
        score = [similarity_scores[i][1] for i in range(len(recommended_products))]  # Use the length of recommendations

        # Pad the recommendations to have a length of 5
        recommended_products += [''] * (100 - len(recommended_products))
        score += [0] * (5 - len(score))

        # Ensure that both lists have the same length
        min_length = min(len(recommended_products), len(score))
        recommended_products = recommended_products[:min_length]
        score = score[:min_length]

        recommendations_df = pd.DataFrame({'Id Encoded': [user_id_encoded] * min_length,
                                           'recommended product': recommended_products,
                                           'score recommendation': score})

        return recommendations_df

    else:
        st.warning("No purchase history found.")
        return None


    
item_matrix = pd.read_csv("amazon.csv")  

def recommend_product(product_data, selected_products):
    product_labels = []
    for product_name in selected_products:
        matches = product_data[product_data["product_name"] == product_name]
        if not matches.empty:
            product_labels.append(matches["product_name_label"].values[0])

    recommended_products = []
    for label in product_labels:
        label_str = str(label)
        recommended_products.append(list(item_matrix.loc[label_str].sort_values(ascending=False).iloc[:20].index))

    recommendations_info = []
    for recommendations, product_name in zip(recommended_products, selected_products):
        for product_label in recommendations:
            # Replace the comments with your actual code to collect product information
            product_info = {
                'Recommended Product Label': product_label,
                'Recommended Product Name': product_data.loc[product_data['product_name_label'] == int(product_label), 'product_name'].values[0],
                'Other Information': product_data.loc[product_data['product_name_label'] == int(product_label), 'other_column'].values[0],
                # Add more columns as needed
            }
            recommendations_info.append(product_info)

    recommendations_df = pd.DataFrame(recommendations_info)
    return recommendations_df



def zero_preprocessing(cls_data):
    cls_data1=cls_data.copy()
    zero=[]
    for z in cls_data1.columns:
        value=((cls_data1[z]==0).mean()*100).round(2)
        zero.append(value)
    zero_df=pd.DataFrame({"Column_name":cls_data1.columns,"Zero_Percentage":zero}).sort_values("Zero_Percentage",ascending=False)
    col_to_rem=["youtube","days_since_last_visit","bounces","totals_newVisits","latest_isTrueDirect",
         "earliest_isTrueDirect","time_latest_visit","time_earliest_visit","device_isMobile","device_browser","device_operatingSystem","last_visitId","latest_visit_id",
        "visitId_threshold","earliest_visit_id","earliest_visit_number","latest_visit_number","days_since_first_visit",
      "earliest_source","latest_source","earliest_medium","latest_medium","earliest_keyword","latest_keyword",
        "device_deviceCategory","channelGrouping","geoNetwork_region","target_date","bounce_rate","historic_session_page","avg_session_time_page","products_array"]
    cls_data1.drop(col_to_rem,axis=1,inplace=True)
    for spar in cls_data1.columns:
        me=cls_data1[spar].mean()
        if spar=="has_converted" or spar=="transactionRevenue":
            continue
        values=[]
        for spar_val in cls_data1[spar].values:
            if spar_val<=0:
                values.append(me)
            else:
                values.append(spar_val)
        cls_data1[spar]=values
    zero1=[]
    for z in cls_data1.columns:
        value=((cls_data1[z]==0).mean()*100).round(2)
        zero1.append(value)
    zero_df_pre=pd.DataFrame({"Column_name":cls_data1.columns,"Zero_Percentage":zero1}).sort_values("Zero_Percentage",ascending=False)
    return zero_df,zero_df_pre,cls_data1   

def outlier(cls_data):
    cls_data2=cls_data.copy()
    for out in cls_data2.columns:
        if out == "transactionRevenue":
            cls_data2[out]=np.log1p(cls_data2[out])
        elif out == "has_converted":
            continue
        else:
            cls_data2[out],_=stats.boxcox(cls_data2[out])
    return cls_data2


def predict(l, tras):
    # Check if the data is constant
    if np.all(l == l[0]):  # If all elements are the same
        print("Warning: Data is constant. Returning original values.")
        return [1]  # Assume 1 for not converted if all values are the same
    
    # Apply Box-Cox transformation
    l, _ = stats.boxcox(l)
    
    # Log1p transformation for tras
    t = np.log1p(tras)
    
    # Prepare input for prediction
    li = list(l) + [t]  # Combine transformed data with t
    
    # Load the model and make predictions
    with open("log_reg.pkl", "rb") as lg_file:
        lg = pickle.load(lg_file)
        
    predicted = lg.predict([li])  # Make the prediction
    return predicted

def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    return log_reg_model

def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    with open("svm_model.pkl", "wb") as file:
        pickle.dump(svm_model, file)

def train_knn(X, y, n_neighbors=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    with open("knn_model.pkl", "wb") as file:
        pickle.dump(knn_model, file)

def model_predict(data, tar, model):
    # Assuming data is a DataFrame or a 2D array-like structure
    # Assuming tar is a 1D array-like structure
    train_data, test_data, train_lab, test_lab = train_test_split(data, tar, test_size=0.2, random_state=42)
    train_pred = model.predict(train_data)
    test_pred = model.predict(test_data)
    
    train_acc = round(accuracy_score(train_lab, train_pred) * 100,2)
    train_prec = round(precision_score(train_lab, train_pred, zero_division=0) * 100,2)
    train_recall = round(recall_score(train_lab, train_pred, zero_division=0) * 100,2)
    train_f1 = round(f1_score(train_lab, train_pred, zero_division=0) * 100,2)
    
    test_acc = round(accuracy_score(test_lab, test_pred) * 100,2)
    test_prec = round(precision_score(test_lab, test_pred, zero_division=0) * 100,2)
    test_recall = round(recall_score(test_lab, test_pred, zero_division=0) * 100,2)
    test_f1score = round(f1_score(test_lab, test_pred, zero_division=0) * 100,2)
    
    return train_acc, train_prec, train_recall, train_f1, test_acc, test_prec, test_recall, test_f1score



def getting_data():
    cls_data=pd.read_csv("classification_data.csv")
    sh=cls_data.shape
    is_null=(cls_data.isnull().mean()*100).round(2)
    is_null_df=pd.DataFrame({"Column_Name":is_null.index,"Null_Percentage":is_null.values}).sort_values("Null_Percentage",ascending=False)
    cls_data.drop_duplicates(inplace=True)
    sh1=cls_data.shape
    des=cls_data.describe()
    zero_df,zerodf_pre,cls_data1=zero_preprocessing(cls_data)
    return sh,sh1,des,is_null_df,zero_df,zerodf_pre,cls_data1


def image_details(img):
    f=img.format
    h=img.size[0]
    w=img.size[1]
    arr=np.array(img)
    arr_size=arr.shape
    m=img.mode
    return f,h,w,arr,arr_size,m

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



with st.sidebar:

    st.sidebar.image("profile.jpg")
    opt = option_menu("Menu",
                      ["INTRO", "Prediction","Image Preprocessing","Text Preprocessing","Recommendation"],menu_icon="cast",styles={"container": {"padding":"4!important"},"nav-link": {"text-align":"left"},"nav-link-selected": {"background-color": "#C2452D"}})
if opt=="INTRO":
    st.markdown("<h2><FONT COLOR='red'>Project Overview</h3>",unsafe_allow_html=True)
    
    st.write("""<p>This project is designed to provide an integrated solution for translating text, recommending products, and processing images in a user-friendly interface. It aims to enhance user interaction through predictive analytics and personalized recommendations while ensuring seamless handling of textual and visual data.</h2></p>""", unsafe_allow_html=True)

    st.write("""<h4><b>1. Translation Module:</h4></n></b><p> This component allows users to input text in one language and receive a translation in another. It supports multiple languages and leverages a machine learning-based translator to ensure accurate and efficient translations.</p>""", unsafe_allow_html=True)

    st.write("""<h4><b>2. Recommendation System:</h4></n></b><p> The recommendation system analyzes user data to suggest relevant products. By employing collaborative filtering techniques and machine learning algorithms, it predicts user preferences based on historical interactions, enhancing the shopping experience.</p>""", unsafe_allow_html=True)

    st.write("""<h4><b>3. Image Preprocessing:</h4></n></b><p> This module focuses on preparing images for analysis and recommendation purposes. It includes techniques like resizing, normalization, and data augmentation to ensure high-quality images that improve the accuracy of predictions.</p>""", unsafe_allow_html=True)

    st.write("""<h4><b>4. Text Preprocessing:</h4></n></b><p> Text preprocessing ensures that input text is clean and structured before it is used in translation or recommendation tasks. This includes tokenization, stop-word removal, and normalization, which enhance the performance of natural language processing tasks.</p>""", unsafe_allow_html=True)

    st.write("""<p>Classification is a type of machine learning task where the goal is to predict the category or class of a given input based on its features. Data preprocessing follows, involving tasks such as handling missing values, encoding categorical variables, and scaling features to ensure the dataset's quality and consistency.</p>""",unsafe_allow_html=True)
    st.write("""<h4><b>Task Definition:</h4><p> Clearly define the classification task, including the classes or categories you want to predict.""",unsafe_allow_html=True)
    st.write("""<h4><b>Data Collection:</h4></b><p> Gather a dataset that includes examples of input features along with their corresponding class labels.""",unsafe_allow_html=True)
    st.write("""<h4><b>Data Preprocessing:</h4></b><p> Clean and preprocess the data. This may involve handling missing values, encoding categorical variables, and scaling numerical features.</p>""",unsafe_allow_html=True)
    st.write("""<h4><b>Feature Extraction:</h4></b><p> Identify and select relevant features from the input data that will be used to train the model.</p>""",unsafe_allow_html=True)
    st.write("""<h4><b>Split the Data:</h4></b><p> Split the dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance on unseen data.</p>""",unsafe_allow_html=True)
    st.write("""<h4><b>Choose a Classification Algorithm:</h4></b><p> Select a classification algorithm suitable for your data and problem. Common algorithms include decision trees, support vector machines, logistic regression, k-nearest neighbors, and neural networks.</p>""",unsafe_allow_html=True)
    st.write("""<h4><b>Train the Model:</h4></b><p> Use the training data to train the classification model. The model learns the patterns and relationships in the data that allow it to make predictions.""",unsafe_allow_html=True)
    st.write("""<h4><b>Hyperparameter Tuning:</h4></b><p> Fine-tune the hyperparameters of your model to optimize its performance. This may involve using techniques like cross-validation.""",unsafe_allow_html=True)
        
        # Load images
    linkedin_image_path = r"C:\Users\skath\OneDrive\Desktop\DTM9\Ecommerce-Customer-Behavior-Final-Project--main\linked.png"
    github_image_path = r"C:\Users\skath\OneDrive\Desktop\DTM9\Ecommerce-Customer-Behavior-Final-Project--main\github.jpeg"

    # Convert images to base64
    linkedin_base64 = image_to_base64(linkedin_image_path)
    github_base64 = image_to_base64(github_image_path)

    # Footer content with images
    footer_connect = f"""
        <div style="background-color: black; padding: 20px; color: white; text-align: center;">
            <h5 style="margin: 0;">CONNECT</h5>
            <h5 style="margin: 0;">
                <a href="https://www.linkedin.com/in/jesima-parvin/" style="text-decoration: none; color: white;">
                    <img src="data:image/png;base64,{linkedin_base64}" alt="LinkedIn" width="30" style="vertical-align: middle;">  <!-- Adjusted width -->
                </a>
                <a href="https://github.com/Jesi1511" style="text-decoration: none; color: white;">
                    <img src="data:image/jpeg;base64,{github_base64}" alt="GitHub" width="30" style="vertical-align: middle;">  <!-- Adjusted width -->
                </a>
                <h5 style ="margin: 0;"> JESIMA PARVIN M</h5>
            </h5>
        </div>
        """


    # Display the footer
    st.markdown(footer_connect, unsafe_allow_html=True)

if opt=="Prediction":
    st.markdown("<h2><FONT COLOR='red'>Predicton</h3>",unsafe_allow_html=True)
    
    shap,shape,des,isnull,zerodf,zerodf_pre,cls_data= getting_data()
    tab1,tab2=st.tabs(["EDA","Prediction"])
    #plt.plot([1, 2, 3], [4, 5, 6])
    #st.pyplot(plt)

    with tab1:
        st.markdown("<h3><FONT COLOR='#000000'>Exploratory Data Analysis (EDA)</h2>",unsafe_allow_html=True)
        st.write("<p>EDA, or Exploratory Data Analysis, is a critical phase in the data analysis process that involves visually and statistically exploring and summarizing key characteristics, patterns, and trends within a dataset. The primary objectives of EDA are to uncover insights, identify relationships, and gain an understanding of the structure and distribution of the data. This process is crucial for informing subsequent steps in the data analysis pipeline, such as feature engineering, modeling, and hypothesis testing.</p>",unsafe_allow_html=True)
        st.markdown("""<h3><FONT COLOR=#000000>Dataset Information</h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4><FONT COLOR=#000000>Shape of Dataset</h4>""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Rows: </b>{shap[0]}""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Columns: </b>{shap[1]}""",unsafe_allow_html=True)
            st.write("""In this 100000 rows there are 90793 duplicates, for model building we can delete all duplicates from the dataset. So that classification will perform better""")
            st.write("""<h4><FONT COLOR=#000000>After Removing all Duplicates: </h5>""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Rows: </b>{shape[0]}""",unsafe_allow_html=True)
            st.write(f"""<p><b>No.of.Columns: </b>{shape[1]}""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4><FONT COLOR=#000000>Aggregate Information</h4>""",unsafe_allow_html=True)
            st.write("""<p>From this table we can take the mean, min, max, etc... for all the numerical value columns</p>""",unsafe_allow_html=True)
            st.dataframe(des)
        st.write(" ")
        st.markdown("""<h3><FONT COLOR='#000000'>Data Preprocessing</h3>""",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("""<h4><FONT COLOR:'#000000'>Null Percentage</h4>""",unsafe_allow_html=True)
            st.write("""<P>From this table there is no empty values. So we don't need to make any changes</p>""",unsafe_allow_html=True)
            st.dataframe(isnull,hide_index=True)
        with c2:
            st.markdown("""<h4><FONT COLOR:'#000000'>Sparsity Data</h4>""",unsafe_allow_html=True)
            st.write("""<p>In this sparsity table the columns having more than 50 percent of zero values we can delete that columns""",unsafe_allow_html=True)
            st.dataframe(zerodf,hide_index=True)
        with c3:
            st.write("")
            st.write("")
            st.write("")
            st.write("""<p><b>Note:</b> The column has_converted is the target columns so we should not make any changes and removed all non-numeric values.</p>""",unsafe_allow_html=True)
            st.dataframe(zerodf_pre,hide_index=True)
        st.markdown("""<h4><FONT COLOR:'#000000'>Outlier Detection: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.write("""<p>  It's an unusually extreme value that lies outside the typical range of values in a dataset. Identifying outliers is important in machine learning</p> """,unsafe_allow_html=True)
            st.plotly_chart(px.box(cls_data))
            st.write("")

        with c2:
            cls_data1=outlier(cls_data)
            st.write("""<p> We can treat the outliers by changing the values using boxcox method. After treating the outlier the mean,median and mode values will be in the plot</p> """,unsafe_allow_html=True)
            st.plotly_chart(px.box(cls_data1))

        st.markdown("""<h4><FONT COLOR:'#000000'>Distributation Curve: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)

        with c1:
            st.write("""<p>This graphs are ploted to show the distribution of values for induvial columns. Most of the columns are Right Skewed this is not normaly distributed</p>""",unsafe_allow_html=True)
            on1 = st.toggle('View Distribution curve',key="on1")

            if on1:
                #st.set_option('deprecation.showPyplotGlobalUse', False)              
                for i in cls_data.columns:
                    #fig, ax = plt.subplot(figsize=(10,6))
                    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
                    sns.set_theme(style="whitegrid")
                    #plt.figure(figsize=(10, 6))
                    #sns.kdeplot(cls_data[i], shade=True, ax=ax)
                    sns.kdeplot(cls_data[i], shade=True, ax=ax)  # Use ax to specify where to draw the plot
                    st.pyplot(fig)  # Pass the figure to st.pyplot

                    #sns.displot(cls_data[i],kind='kde')
                    #st.pyplot()
        with c2:
            st.write("""<p>Here we used boxcox and log1p method to make right skewd graph to normal distribution</p>""",unsafe_allow_html=True)
            on2 = st.toggle('View Distribution curve',key="on2")
            if on2:
                #st.set_option('deprecation.showPyplotGlobalUse', False)
                for i in cls_data.columns:
                    #sns.set_theme(style="whitegrid")
                    #plt.figure(figsize=(10, 6))
                    #sns.displot(cls_data1[i],kind='kde')
                    #st.pyplot()
                    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
                    sns.set_theme(style="whitegrid")  # Updated to set_theme
                    sns.kdeplot(cls_data1[i], shade=True, ax=ax)  # Use ax to specify where to draw the plot
                    st.pyplot(fig)  # Pass the figure to st.pyplot
                
        st.markdown("""<h4><FONT COLOR:'#000000'>Correlation Heatmap: </h4>""",unsafe_allow_html=True)
        
        c1,c2=st.columns(2)
        
        with c1:
            cls_data2=cls_data1.copy()
            cls_data2.drop("has_converted",axis=1,inplace=True)
            st.write("""<p>Correlation heatmap for all continous variables that quantifies the degree to which two variables are related </p>""",unsafe_allow_html=True)
            st.write("")

            corr_data = cls_data2.corr()
            #st.set_option('deprecation.showPyplotGlobalUse', False)
            #sns.set_theme(style="whitegrid")
            #plt.figure(figsize=(15,6))
            #plt.title("Correlation Heatmap")
            #sns.heatmap(corr_data,annot=True,cmap="coolwarm",fmt=".3f")
            #st.pyplot()
            fig, ax = plt.subplots(figsize=(15, 6))  # Create figure and axes
            sns.set_theme(style="whitegrid")  # Updated to set_theme
            sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".3f", ax=ax)  # Use ax to specify where to draw the plot
            st.pyplot(fig)  # Pass the figure to st.pyplot
            
        with c2:
            cls_data2.drop(["count_session","count_hit","historic_session","single_page_rate"],axis=1,inplace=True)
            st.write("""<p> From the previous correlation map the columns <b>count session,count hit, historic_session, single_page_rate</b> are having highest correaltion, we can remove that columns</p>""",unsafe_allow_html=True)
            corr_data = cls_data2.corr()

            #st.set_option('deprecation.showPyplotGlobalUse', False)
            #sns.set_theme(style="whitegrid")
            #plt.figure(figsize=(15,6))
            #plt.title("Correlation Heatmap")
            #sns.heatmap(corr_data,annot=True,cmap="coolwarm",fmt=".3f")
            #st.pyplot()
            fig, ax = plt.subplots(figsize=(15, 6))  # Create figure and axes
            sns.set_theme(style="whitegrid")  # Updated to set_theme
            sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".3f", ax=ax)  # Use ax to specify where to draw the plot
            st.pyplot(fig)  # Pass the figure to st.pyplot
            
        st.markdown("""<h4><FONT COLOR:'#000000'>Feature Importance: </h4>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ran_class=RandomForestClassifier(n_estimators=20,random_state=44)
            over_data=cls_data2
            over_tar=cls_data1['has_converted']
            ran_class.fit(over_data,over_tar)
            val=ran_class.feature_importances_*100
            feature_df=pd.DataFrame({"Columns":over_data.columns,"Feature_percentage":val}).sort_values("Feature_percentage",ascending=False)
            st.dataframe(feature_df)
            over_data.drop(["sessionQualityDim","avg_visit_time","geoNetwork_longitude","geoNetwork_latitude"],axis=1,inplace=True)
        with c2:
            st.write("""<p>In this Feature Importance we can understand the importance of the columns in the dataset.</p>""",unsafe_allow_html=True)
            st.write("""<p>We can remove the sessionqualityDim, geonetwork latitude, geonetwork longititude, avg_"visit_time</p>""",unsafe_allow_html=True)
            
    with tab2:
        st.write("""<h3>Prediction</h4>""",unsafe_allow_html=True)
        st.write("""<p><b><i>NOTE:</i> </b>Kindly enter only positive values and dont't enter negative value or zero.""",unsafe_allow_html=True)
        c1,c2,c3,c4,c5=st.columns(5)
        with c1:
            ses = st.number_input("Average Session Value", min_value=0.0)  # Ensure positive value
        with c2:
            visit = st.number_input("Visits Per Day", min_value=0)  # Ensure positive value
        with c3:
            inter = st.number_input("Number of Interactions", min_value=0)  # Ensure positive value
        with c4:
            # Use time_input for time on site
            time = st.time_input("Time on Site (hh:mm)", value=datetime.time(0, 0))  # Default to 00:00
        with c5:
            tras = st.number_input("Transaction Revenue", min_value=0.0)  # Ensure positive value

        if st.button("Predict", key="bt1"):
            # Convert the time input to total minutes or seconds for further processing
            total_seconds = time.hour * 3600 + time.minute * 60

            # Decision logic based on the specified conditions
            if tras > 0:
                converted = True
            elif ses > 100 and visit > 5:
                converted = True
            else:
                converted = False

            # Display the conversion status based on the decision logic
            if converted:
                st.write("""<h4><FONT COLOR='green'>User Converted</h4>""", unsafe_allow_html=True)
            else:
                st.write("""<h4><FONT COLOR='red'>User Not Converted</b></h4>""", unsafe_allow_html=True)

        st.write("")
        st.write("")
        
        st.markdown("""<h3>Model Result</h3>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("""<h4>Logistic Regression</h3>""", unsafe_allow_html=True)
            st.write("""<p>Logistic regression is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome. </p>""", unsafe_allow_html=True)
            st.write("""<p>Logistic regression is a useful analysis method for classification problems, where you are trying to determine if a new sample fits best into a category. Aspects of cybersecurity are classification problems, such as attack detection, and logistic regression is a useful analytic technique.</p>""", unsafe_allow_html=True)
        
        with c2:
            st.markdown("""<h4>Metrics Score</h4>""", unsafe_allow_html=True)
            X = np.random.rand(100, 5)  # Features
            y = np.random.choice([0, 1], size=100)  # Labels
            
            # Train models
            log_reg_model = train_logistic_regression(X, y)
            train_svm(X, y)
            train_knn(X, y)

            
            with open("log_reg_model.pkl", "wb") as file:
                    pickle.dump(log_reg_model, file)
                
            with open("log_reg_model.pkl", "rb") as lg:
                    log_reg_model = pickle.load(lg)

            train_acc, train_prec, train_recall, train_f1, test_acc, test_prec, test_recall, test_f1score = model_predict(X, y, log_reg_model)
            log_reg_df = pd.DataFrame({
                "Accuracy": [train_acc, test_acc],
                "Precision": [train_prec, test_prec],
                "Recall": [train_recall, test_recall],
                "F1 Score": [train_f1, test_f1score]
            }, index=["Training Score", "Testing Score"])
            
            print("Logistic Regression:")
            print(f"Training Accuracy: {train_acc}%")
            print(f"Testing Accuracy: {test_acc}%")
            st.dataframe(log_reg_df)


             # Create a bar chart for visual representation of metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            train_scores = log_reg_df.loc['Training Score'].values
            test_scores = log_reg_df.loc['Testing Score'].values

            # Plotting the bar chart
            fig, ax = plt.subplots()
            width = 0.35 # Width of the bars
            x = range(len(metrics))

            # Create bars
            bars1 = ax.bar(x, train_scores, width, label='Training Score', color='b')
            bars2 = ax.bar([p + width for p in x], test_scores, width, label='Testing Score', color='r')

            # Adding labels and title
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Scores')
            ax.set_title('Logistic Regression Model Performance Metrics')
            ax.set_xticks([p + width / 2 for p in x])
            ax.set_xticklabels(metrics)
            ax.legend()

        # Display the chart in Streamlit
        st.pyplot(fig)

        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4>Support Vector Machine</h3>""",unsafe_allow_html=True)
            st.write("""<p>A support vector machine (SVM) is a type of supervised learning algorithm used in machine learning to solve classification and regression tasks; SVMs are particularly good at solving binary classification problems, which require classifying the elements of a data set into two groups.</p>""",unsafe_allow_html=True)
            st.write("""<p>The sigmoid kernel is widely applied in neural networks for classification processes. The SVM classification with the sigmoid kernel has a complex structure and it is difficult for humans to interpret and understand how the sigmoid kernel makes classification decisions.</p>""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4>Metrics Score</h4>""",unsafe_allow_html=True)
            
            with open("svm_model.pkl", "rb") as sv:
                svm_model = pickle.load(sv)
                
            train_acc, train_prec, train_recall, train_f1, test_acc, test_prec, test_recall, test_f1score = model_predict(X, y, svm_model)
            svm_df = pd.DataFrame({
                "Accuracy": [train_acc, test_acc],
                "Precision": [train_prec, test_prec],
                "Recall": [train_recall, test_recall],
                "F1 Score": [train_f1, test_f1score]
            }, index=["Training Score", "Testing Score"])
            
            print("\nSupport Vector Machine:")
            print(f"Training Accuracy: {train_acc}%")
            print(f"Testing Accuracy: {test_acc}%")
            st.dataframe(svm_df)


             # Create a bar chart for visual representation of metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            train_scores = svm_df.loc['Training Score'].values
            test_scores = svm_df.loc['Testing Score'].values

            # Plotting the bar chart
            fig, ax = plt.subplots()
            width = 0.35  # Width of the bars
            x = range(len(metrics))

            # Create bars
            bars1 = ax.bar(x, train_scores, width, label='Training Score', color='b')
            bars2 = ax.bar([p + width for p in x], test_scores, width, label='Testing Score', color='r')

            # Adding labels and title
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Scores')
            ax.set_title('SVM Model Performance Metrics')
            ax.set_xticks([p + width / 2 for p in x])
            ax.set_xticklabels(metrics)
            ax.legend()

            # Display the chart in Streamlit
        st.pyplot(fig)

        c1,c2=st.columns(2)
        with c1:
            st.markdown("""<h4>KNN Algorithm</h3>""",unsafe_allow_html=True)
            st.write("""<p>The k-nearest neighbors algorithm, is a non-parametric, supervised learning classifier, which uses proximity to make classifications or predictions about the grouping of an individual data point.</p>""",unsafe_allow_html=True)
            st.write("""<p>For classification problems, a class label is assigned on the basis of a majority vote—i.e. the label that is most frequently represented around a given data point is used. While this is technically considered “plurality voting”, the term, “majority vote” is more commonly used in literature.</p>""",unsafe_allow_html=True)
        with c2:
            st.markdown("""<h4>Metrics Score</h3>""",unsafe_allow_html=True)
            with open("knn_model.pkl", "rb") as knn:
                knn_model = pickle.load(knn)

            train_acc, train_prec, train_recall, train_f1, test_acc, test_prec, test_recall, test_f1score = model_predict(X, y, knn_model)
            knn_df = pd.DataFrame({
                "Accuracy": [train_acc, test_acc],
                "Precision": [train_prec, test_prec],
                "Recall": [train_recall, test_recall],
                "F1 Score": [train_f1, test_f1score]
            }, index=["Training Score", "Testing Score"])
            
            print("\nK-Nearest Neighbors:")
            print(f"Training Accuracy: {train_acc}%")
            print(f"Testing Accuracy: {test_acc}%")
            st.dataframe(knn_df)

            # Create a bar chart for visual representation of metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            train_scores = knn_df.loc['Training Score'].values
            test_scores = knn_df.loc['Testing Score'].values
            
            # Plotting the bar chart
            fig, ax = plt.subplots()
            width = 0.35  # Width of the bars
            x = range(len(metrics))

            # Create bars
            bars1 = ax.bar(x, train_scores, width, label='Training Score', color='b')
            bars2 = ax.bar([p + width for p in x], test_scores, width, label='Testing Score', color='r')

            # Adding labels and title
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Scores')
            ax.set_title('KNN Model Performance Metrics')
            ax.set_xticks([p + width / 2 for p in x])
            ax.set_xticklabels(metrics)
            ax.legend()

        # Display the chart in Streamlit
        st.pyplot(fig)
                        
if opt=="Image Preprocessing":
    st.markdown("""<h3><FONT COLOR:'red'>Image Preprocessing</h3>""",unsafe_allow_html=True)
    select = option_menu(None,["About","Preprocess"],orientation="horizontal",key="image_side")
    if select=="About":
        st.write("""<p>Image preprocessing is a crucial step in computer vision and image analysis tasks. It involves applying various techniques to enhance the quality of images, reduce noise, and prepare them for further analysis.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Grayscale Conversion: </b>A grayscale image is a type of digital image where each pixel is represented by a single intensity value, typically ranging from 0 to 255, with 0 being black and 255 being white.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Resizing: </b>Image resizing is a fundamental operation in digital image processing, involving the adjustment of the dimensions of an image. This process is commonly applied to adapt images for specific purposes, such as fitting them into predefined spaces, reducing file sizes, or preparing them for analysis by machine learning models.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Bluring Image: </b>Blurring is a common image processing technique that involves the reduction of image details, resulting in a smoothed or softened appearance. This operation is widely used for various purposes, such as noise reduction, privacy protection, and aesthetic enhancements in photography and computer vision applications.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Contrast Image: </b>Contrast enhancement is a fundamental image processing technique employed to improve the perceptual distinction between the light and dark regions within an image. This method plays a crucial role in enhancing the visual quality and interpretability of digital images across a broad spectrum of applications.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Edge Detection: </b>Edge detection is a fundamental image processing technique designed to identify boundaries within an image, where significant changes in intensity or color occur. These boundaries often represent transitions between different objects or regions in the scene, making edge detection a crucial step in various computer vision and image analysis applications.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Sharping Image: </b>Image sharpening is a key image processing technique employed to enhance the fine details and edges within an image, resulting in a more visually striking and refined appearance. The goal of sharpening is to emphasize transitions in pixel intensities, making edges and fine structures more pronounced.</p>""",unsafe_allow_html=True)
        st.write("""<p><b>Negative Image: </b>Creating a negative image is a basic but impactful image processing technique that involves inverting the pixel intensities of a photograph. In a negative image, the colors are transformed such that bright areas become dark, and vice versa. This transformation is achieved by subtracting each pixel value from the maximum possible intensity value.</p>""",unsafe_allow_html=True)
    if select=="Preprocess":
        st.write("""<h5>Select any image</h5>""",unsafe_allow_html=True)
        input_file=st.file_uploader("Select an image: ", type=["jpg", "jpeg", "png"],key="upload1")
        if input_file is None:
            st.write("<p><b>Please select your Image</b></p>",unsafe_allow_html=True)
        else:
            image = Image.open(input_file)
            st.write(" ")
            st.markdown("""<h4>Original Image</h3>""",unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1:
                st.image(image)
            with c2:
                fr,hi,wi,arr,ar_s,mo=image_details(image)
                st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                st.write("<b>Image RGB Array colors:</b>",[arr],unsafe_allow_html=True)
                st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
            gray_image = image.convert("L")
            st.write(" ")

            st.write("""<h5>Select any types of preprocessing techniques:</h5>""",unsafe_allow_html=True)
            processing=st.multiselect("Choose any preprocessing techniques:",["Gray Scale","Color Adjustment","Color Change","Resizing Image","Bluring Image","Contrast Image","Edge Detection","Sharping Image","Negative Image","Brightness","Text in Image"])
            for i in processing:
                if i=="Gray Scale":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Gray Scale Conversation</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        gray_image = image.convert("L")
                        st.image(gray_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(gray_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Resizing Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Resizing Image:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        h=st.slider("Choose size", 0, 5000,400)
                        w=st.slider("Choose Width:",0,5000,200)
                        resized_image = gray_image.resize((h,w))
                        st.image(resized_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(resized_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Bluring Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Bluring Image:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        b=st.slider("Choose size", 0, 100,3)
                        blur_image = gray_image.filter(ImageFilter.GaussianBlur(radius=b)) 
                        st.image(blur_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(blur_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Contrast Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Contrast Image</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        c=st.slider("Choose value:",0,255,10)
                        process_1 = ImageEnhance.Contrast(gray_image)
                        process=process_1.enhance(c)
                        st.image(process)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(process)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Edge Detection":
                    st.write(" ")
                    st.markdown("""<h3>Edge detection</h3>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        g=st.slider("Choose value",0,20,5)
                        gray_edge = gray_image.filter(ImageFilter.FIND_EDGES)
                        edge_bright = ImageEnhance.Brightness(gray_edge)
                        edge_ = edge_bright.enhance(g)
                        st.image(edge_)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(edge_)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Sharping Image":
                    st.write(" ")
                    st.markdown("""<h3>Sharping Image:<h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        s=st.slider("Choose value",0,30,5)
                        sharp_img = ImageEnhance.Sharpness(image)
                        sharp=sharp_img.enhance(s)
                        st.image(sharp)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(sharp)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Brightness":
                    st.write(" ")
                    st.markdown("""<h3>Brightness:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        br=st.slider("Choose value",0,100,10)
                        edge_bright = ImageEnhance.Brightness(gray_image)
                        bright = edge_bright.enhance(br)
                        st.image(bright)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(bright)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Negative Image":
                    st.write(" ")
                    st.markdown("""<h3><FONT COLOR:#000000>Negative Image:</h4>""",unsafe_allow_html=True)
                    c1,c2=st.columns(2)
                    with c1:
                        neg_image = ImageOps.invert(gray_image)
                        st.image(neg_image)
                    with c2:
                        fr,hi,wi,arr,ar_s,mo=image_details(neg_image)
                        st.write("<b>Image format:</b>",fr,unsafe_allow_html=True)
                        st.write("""<b>Image Mode:</b>""",mo,unsafe_allow_html=True)
                        st.write("<b>Height:</b>",hi,unsafe_allow_html=True)
                        st.write("<b>Width:</b>",wi,unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>",[arr],unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>",ar_s,unsafe_allow_html=True)
                if i=="Text in Image":
                    st.write(" ")
                    st.markdown("""<h3>Text in Image</h4>""",unsafe_allow_html=True)
                    reader = easyocr.Reader(['en'])
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result = reader.readtext(opencv_image,detail=0)
                    c1,c2=st.columns(2)
                    with c1:
                        st.image(image)
                    with c2:
                        if len(result)==0:
                            st.write("")
                            st.write("")
                            st.write("""<b>There is no text in the given image...</b>""",unsafe_allow_html=True)
                        else:
                            st.write("""Text from the Image""")
                            st.write(result)
                if i == "Color Adjustment":
                    st.write(" ")
                    st.markdown("""<h3>Color Adjustment</h3>""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        r = st.slider("Adjust Red Channel (0-2)", 0.0, 2.0, 1.0)
                        g = st.slider("Adjust Green Channel (0-2)", 0.0, 2.0, 1.0)
                        b = st.slider("Adjust Blue Channel (0-2)", 0.0, 2.0, 1.0)
                        color_adjusted_image = ImageEnhance.Color(image).enhance(r)
                        color_adjusted_image = ImageEnhance.Color(color_adjusted_image).enhance(g)
                        color_adjusted_image = ImageEnhance.Color(color_adjusted_image).enhance(b)
                        st.image(color_adjusted_image)
                    with c2:
                        fr, hi, wi, arr, ar_s, mo = image_details(color_adjusted_image)
                        st.write("<b>Image format:</b>", fr, unsafe_allow_html=True)
                        st.write("<b>Image Mode:</b>", mo, unsafe_allow_html=True)
                        st.write("<b>Height:</b>", hi, unsafe_allow_html=True)
                        st.write("<b>Width:</b>", wi, unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>", [arr], unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>", ar_s, unsafe_allow_html=True)

                if i == "Color Change":
                    st.write(" ")
                    st.markdown("""<h3>Color Change</h3>""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        # Color Picker to allow users to choose any color
                        selected_color = st.color_picker("Pick a color", "#FF0000")  # Default color is red

                        # Convert the original image to grayscale first
                        gray_image = image.convert("L")
                        
                        # Colorize the grayscale image with the selected color
                        color_image = ImageOps.colorize(gray_image, black="black", white=selected_color)
                        
                        # Display the colorized image
                        st.image(color_image)

                    with c2:
                        fr, hi, wi, arr, ar_s, mo = image_details(color_image)
                        st.write("<b>Image format:</b>", fr, unsafe_allow_html=True)
                        st.write("<b>Image Mode:</b>", mo, unsafe_allow_html=True)
                        st.write("<b>Height:</b>", hi, unsafe_allow_html=True)
                        st.write("<b>Width:</b>", wi, unsafe_allow_html=True)
                        st.write("<b>Image 2D Array colors:</b>", [arr], unsafe_allow_html=True)
                        st.write("<b>Array Image Shape:</b>", ar_s, unsafe_allow_html=True)
                        



if opt == "Text Preprocessing":
    st.markdown("""<h3><font color='red'>Text Preprocessing</h3>""", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Text preprocess", "Translate"])
    # Initialize tokenizer and other tools
    tokenizer = TreebankWordTokenizer()
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    with tab1:
        o = st.radio("Select one option", options=["Browse image", "Enter text"])
        
        if o == "Browse image":
            input_file = st.file_uploader("Select an image: ", type=["jpg", "jpeg", "png"], key="upload2")
            if input_file is None:
                st.write("""<p><b>Please select the image</b></p>""", unsafe_allow_html=True)
            else:
                image = Image.open(input_file)
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Perform OCR using easyocr
                reader = easyocr.Reader(['en'])
                result = reader.readtext(opencv_image)
                text_input = ' '.join([entry[1] for entry in result])
                
                # Now you can use 'text_input' in further processing
                st.write(f"Extracted Text: {text_input}")

        if o == "Enter text":
            st.write("""<p>You can enter any text here:</p>""", unsafe_allow_html=True)
            text_input = st.text_area("Enter your text:")

        text_opt_select = st.multiselect("Select any preprocessing techniques:", 
                                         ["Word Tokenize", "Stop Word Removal", 
                                          "Removing Special Characters", "Stemming", 
                                          "Lemmatizer", "Tokenization", "Word Frequency", 
                                          "Remove Punctuation", "Lowercasing", 
                                          "Word Cloud", "Sentiment Analysis Score", 
                                          "Keyword Extraction"])

        if st.button("Process"):
            # Tokenization using the TreebankWordTokenizer
            sample_tokens = tokenizer.tokenize(text_input)  # Correctly call the tokenize method
            
            # Removing special characters using regex
            spl = [re.sub(r'[^a-zA-Z0-9]', '', i) for i in sample_tokens]
            
            for i in text_opt_select:
                if i == "Word Tokenize":
                    st.markdown("""<h4>Word Tokenize</h4>Word Tokenization is the process of splitting text into individual words or tokens. This is an essential step in natural language processing (NLP) as it allows the analysis of the text at the word level. For example, the sentence 'Hello world!' would be tokenized into ['Hello', 'world']. """, unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(" ")
                        st.write("""<h5>Original Text</h5>""", unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Word Tokenize</h5>""", unsafe_allow_html=True)
                        st.write(sample_tokens)
                        tokens_df = pd.DataFrame(sample_tokens, columns=["Tokens"])
                        st.dataframe(tokens_df) 

                if i == "Stop Word Removal":
                    st.markdown("""<h4>Stop Word Removal</h4>Stop Word Removal is a common preprocessing step in natural language processing (NLP) that involves filtering out words that are considered to be of little value in terms of their meaning or contribution to the overall understanding of the text. Common stop words include 'and', 'the', 'is', etc. For instance, from the sentence 'The cat sat on the mat', the remaining words would be ['cat', 'sat', 'mat'].""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(" ")

                        st.write("""<h5>Original Text</h5>""", unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Stop Word Removal</h5>""", unsafe_allow_html=True)
                        stp_rem = [i for i in sample_tokens if i.lower() not in stop_words]
                        st.write(stp_rem)
                        filtered_tokens_df = pd.DataFrame(stp_rem, columns=["Filtered Tokens"])
                        st.dataframe(filtered_tokens_df)

                if i == "Removing Special Characters":
                    st.markdown("""<h4>Removing Special Characters</h4>This technique involves eliminating special characters (such as punctuation and symbols) from the text. Special characters do not usually contribute to the semantic meaning and can interfere with text analysis. For example, the phrase 'Hello, world!' would be cleaned to 'Hello world'.""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(" ")
                        st.write("""<h5>Original Text</h5>""", unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Removing Special Characters</h5>""", unsafe_allow_html=True)
                        st.write(spl)

                if i == "Stemming":
                    st.markdown("""<h4>Stemming</h4>Stemming is the process of reducing a word to its root form. This is helpful in simplifying different forms of a word to a common base form. For example, 'running', 'runner', and 'ran' would all be reduced to 'run'.""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(" ")
                        st.write("""<h5>Original Text</h5>""", unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Stemming</h5>""", unsafe_allow_html=True)
                        stem = [stemmer.stem(i) for i in sample_tokens]
                        st.write(stem)

                if i == "Lemmatizer":
                    st.markdown("""<h4>Lemmatizer</h4>Lemmatization is similar to stemming but considers the context of the word, converting it to its base or dictionary form. For example, 'better' would be converted to 'good', which is its lemma.""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(" ")
                        st.write("""<h5>Original Text</h5>""", unsafe_allow_html=True)
                        st.write(text_input)
                    with c2:
                        st.write("""<h5>Lemmatizer</h5>""", unsafe_allow_html=True)
                        lem = [lemmatizer.lemmatize(i) for i in sample_tokens]
                        st.write(lem)

                if i == "Tokenization":
                    st.markdown("""<h4>Tokenization</h4>Tokenization is another way of splitting text into tokens, similar to Word Tokenization. This allows for more granular text analysis at the token level.""", unsafe_allow_html=True)
                    tokens = tokenizer.tokenize(text_input)
                    st.write(tokens)

                if i == "Word Frequency":
                    st.markdown("""<h4>Word Frequency</h4>Word Frequency counts the number of occurrences of each word in a given text. This technique is valuable for understanding which words are most prominent in the text. For example, in the sentence 'cat cat dog', the frequency count would yield {'cat': 2, 'dog': 1}.""", unsafe_allow_html=True)

                    word_freq = Counter(sample_tokens)
                    freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"])
                    st.dataframe(freq_df)

                if i == "Remove Punctuation":
                    st.markdown("""<h4>Remove Punctuation</h4>Removing punctuation marks is important in text preprocessing, as punctuation does not contribute to the meaning of words. For instance, 'Hello, world!' would be processed to 'Hello world'.""", unsafe_allow_html=True)
                    cleaned_text = text_input.translate(str.maketrans('', '', string.punctuation))
                    st.write(cleaned_text)

                if i == "Lowercasing":
                    st.markdown("""<h4>Lowercasing</h4>Lowercasing transforms all characters in the text to lowercase. This step helps in reducing the complexity of the text analysis and ensures uniformity. For example, 'Hello World' becomes 'hello world'.""", unsafe_allow_html=True)
                    lowercased_text = text_input.lower()
                    st.write(lowercased_text)

                if i == "Word Cloud":
                    st.markdown("""<h4>Word Cloud</h4>A Word Cloud visually represents the frequency of words in a given text. Words that appear more frequently are displayed in a larger font, providing an immediate visual cue of the text's most significant terms.""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(" ")
                        st.write("""<h5>Original Text</h5>""", unsafe_allow_html=True)

                        st.write(text_input)
                    with c2:
                        paragraphs = text_input.split('\n\n')  
                        for idx, paragraph in enumerate(paragraphs):
                            st.write(f"""<h5>Word Cloud - Paragraph {idx + 1}</h5>""", unsafe_allow_html=True)

                            # Generate Word Cloud for the current paragraph
                            w_c = WordCloud(width=1000, height=500, background_color="orange").generate(paragraph)

                            # Create the figure and axes
                            fig, ax = plt.subplots(figsize=(14, 6))
                            ax.imshow(w_c)
                            ax.axis('off')  # Turn off axis

                            # Display the figure in Streamlit
                            st.pyplot(fig)

                if i == "Sentiment Analysis Score":
                    st.markdown("""<h4>Sentiment Analysis Score</h4>Sentiment Analysis evaluates the emotional tone of a text, categorizing it as positive, negative, or neutral. This technique is useful for understanding the overall sentiment conveyed in the text. For example, the sentence 'I love this!' would likely yield a positive sentiment score.""", unsafe_allow_html=True)
                    sent_ment = SentimentIntensityAnalyzer()
                    sent_score = sent_ment.polarity_scores(text_input)
                    st.write(sent_score)
                    sc = list(sent_score.values())  # Convert dict_values to a list for plotting
                    fig = px.histogram(x=sc, color=sent_score)
                    fig.update_layout(bargap=0.1)
                    st.plotly_chart(fig)

                if i == "Keyword Extraction":
                    st.markdown("""<h4>Keyword Extraction</h4>Keyword Extraction identifies the most important words or phrases in a text. This technique is beneficial for summarizing the content and understanding its main topics. For example, from the phrase 'Natural Language Processing is fascinating', the extracted keywords could be ['Natural', 'Language', 'Processing', 'fascinating'].""", unsafe_allow_html=True)
                    vect = CountVectorizer(stop_words="english")
                    ext_val = vect.fit_transform([text_input])
                    nf = vect.get_feature_names_out()
                    st.write(nf)   

    with tab2:
            st.markdown("""<h3>Translation</h3>""", unsafe_allow_html=True)

            lan = {
            'af': 'afrikaans', 'sq': 'albanian', 'am': 'amharic', 'ar': 'arabic',
            'hy': 'armenian', 'az': 'azerbaijani', 'eu': 'basque', 'be': 'belarusian',
            'bn': 'bengali', 'bs': 'bosnian', 'bg': 'bulgarian', 'ca': 'catalan',
            'ceb': 'cebuano', 'ny': 'chichewa', 'zh-cn': 'chinese (simplified)',
            'zh-tw': 'chinese (traditional)', 'co': 'corsican', 'hr': 'croatian',
            'cs': 'czech', 'da': 'danish', 'nl': 'dutch', 'en': 'english',
            'eo': 'esperanto', 'et': 'estonian', 'tl': 'filipino', 'fi': 'finnish',
            'fr': 'french', 'fy': 'frisian', 'gl': 'galician', 'ka': 'georgian',
            'de': 'german', 'el': 'greek', 'gu': 'gujarati', 'ht': 'haitian creole',
            'ha': 'hausa', 'haw': 'hawaiian', 'iw': 'hebrew', 'hi': 'hindi',
            'hmn': 'hmong', 'hu': 'hungarian', 'is': 'icelandic', 'ig': 'igbo',
            'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese',
            'jw': 'javanese', 'kn': 'kannada', 'kk': 'kazakh', 'km': 'khmer',
            'ko': 'korean', 'ku': 'kurdish (kurmanji)', 'ky': 'kyrgyz', 'lo': 'lao',
            'la': 'latin', 'lv': 'latvian', 'lt': 'lithuanian', 'lb': 'luxembourgish',
            'mk': 'macedonian', 'mg': 'malagasy', 'ms': 'malay', 'ml': 'malayalam',
            'mt': 'maltese', 'mi': 'maori', 'mr': 'marathi', 'mn': 'mongolian',
            'my': 'myanmar (burmese)', 'ne': 'nepali', 'no': 'norwegian', 'ps': 'pashto',
            'fa': 'persian', 'pl': 'polish', 'pt': 'portuguese', 'pa': 'punjabi',
            'ro': 'romanian', 'ru': 'russian', 'sm': 'samoan', 'gd': 'scots gaelic',
            'sr': 'serbian', 'st': 'sesotho', 'sn': 'shona', 'sd': 'sindhi',
            'si': 'sinhala', 'sk': 'slovak', 'sl': 'slovenian', 'so': 'somali',
            'es': 'spanish', 'su': 'sundanese', 'sw': 'swahili', 'sv': 'swedish',
            'tg': 'tajik', 'ta': 'tamil', 'te': 'telugu', 'th': 'thai', 'tr': 'turkish',
            'uk': 'ukrainian', 'ur': 'urdu', 'uz': 'uzbek', 'vi': 'vietnamese',
            'cy': 'welsh', 'xh': 'xhosa', 'yi': 'yiddish', 'yo': 'yoruba', 'zu': 'zulu',
            'fil': 'Filipino', 'he': 'Hebrew'
        }
        
            src_lang = list(lan.values())
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("""<h4>From:</h4>""", unsafe_allow_html=True)
                source = st.selectbox("Select the source language", src_lang)
                source_lang = list(lan.keys())[list(lan.values()).index(source)]

            with c2:
                st.markdown("""<h4>To:</h4>""", unsafe_allow_html=True)
                des = st.selectbox("Select the destination language", src_lang)
                des_lang = list(lan.keys())[list(lan.values()).index(des)]

            text_translate = st.text_input("Enter your text to translate:")

            if st.button("Translate", key="tras"):
                if text_translate:  # Ensure text input is not empty
                    try:
                        translator = Translator()  # Initialize Translator without parameters
                        
                        # Perform translation
                        translated = translator.translate(text_translate, src=source_lang, dest=des_lang)
                        
                        # Output translated sentence
                        st.write(f"""<h5>Translated Sentence ({lan[des_lang]}):</h5> {translated.text}""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"An error occurred during translation: {str(e)}")
                else:
                    st.warning("Please enter some text to translate.")
if opt == "Recommendation":
    st.markdown("""<h3><font color='red'>Product Recommendation</h3>""", unsafe_allow_html=True)
    
    # Create tabs for "ABOUT" and "Recommend"
    tab1, tab2 = st.tabs(["ABOUT", "Recommend"])
    
    # About Amazon
    with tab1:
        st.markdown("""<h3>History</h3>""", unsafe_allow_html=True)
        st.write("""<p>Amazon was founded in 1994 by Jeff Bezos in Seattle, Washington. It started as an online bookstore but rapidly expanded its product offerings, becoming the world's largest online retailer. Over the years, Amazon has played a pivotal role in revolutionizing e-commerce and shaping the way people shop online.</p>""", unsafe_allow_html=True)

        st.write("""<p>In recent years, Amazon has diversified its business, venturing into various sectors such as cloud computing, artificial intelligence, entertainment, and more. The company's commitment to customer satisfaction, innovative technologies, and rapid delivery services has contributed to its significant growth and global influence.</p>""", unsafe_allow_html=True)
        amazon_link = """
<a href="https://www.amazon.in/" target="_blank">
    <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" alt="Amazon Logo" width="150">
</a>
"""     
        st.markdown(amazon_link, unsafe_allow_html=True)
        st.markdown("""<h3>Products and Services</h3>""", unsafe_allow_html=True)
        st.write("""<p>Amazon offers a wide range of products and services, including but not limited to:</p>""", unsafe_allow_html=True)
        st.write("""<ul>
            <li>Online retail: A vast marketplace where customers can purchase goods across numerous categories.</li>
            <li>Amazon Web Services (AWS): A leading cloud computing platform providing services such as computing power, storage, and databases.</li>
            <li>Amazon Prime: A subscription service offering benefits like fast shipping, streaming of movies and TV shows, and exclusive deals.</li>
            <li>Amazon Echo and Alexa: Smart devices and virtual assistant technology for smart homes.</li>
            <li>Amazon Studios: Producing and streaming original movies and TV shows.</li>
        </ul>""", unsafe_allow_html=True)

        st.markdown("""<h3>Corporate Impact</h3>""", unsafe_allow_html=True)
        st.write("""<p>Amazon's corporate impact extends beyond commerce. The company has been influential in shaping technological advancements, contributing to the growth of e-commerce globally, and playing a key role in the development of cloud computing services. However, Amazon has also faced scrutiny regarding issues such as labor practices, antitrust concerns, and environmental impact.</p>""", unsafe_allow_html=True)

        st.markdown("""<h3>Innovation and Technology</h3>""", unsafe_allow_html=True)
        st.write("""<p>Known for its innovative practices, Amazon has introduced technologies such as drone delivery, cashier-less stores (Amazon Go), and voice-controlled devices. The company continues to invest in research and development, pushing the boundaries of what's possible in the world of technology and e-commerce.</p>""", unsafe_allow_html=True)
        st.write(" ")
        st.write(" ")
        st.markdown("<h3>Partners</h3>", unsafe_allow_html=True)
        st.write("<p>Amazon collaborates with various entities, forming partnerships with:</p>", unsafe_allow_html=True)
        st.write("<ul><li>Brands and Retailers: Amazon provides a platform for brands and retailers to sell their products through the Amazon marketplace.</li><li>Third-Party Sellers: Independent sellers and businesses can leverage Amazon's platform to reach a global customer base.</li><li>Developers: Amazon Web Services (AWS) offers cloud computing services, enabling developers to build and deploy applications.</li><li>Content Creators: Amazon Prime Video and Kindle Direct Publishing support content creators in reaching audiences worldwide.</li></ul>", unsafe_allow_html=True)
        
        # Footer content with links
        footer_content = """
        <div style="background-color: black; padding: 20px; color: white; text-align: center;">
            <a href="https://www.amazon.in/news/" target="_blank" style="text-decoration: none; color: white; margin: 0 15px;">Amazon News</a> | 
            <a href="https://www.amazon.com/pr/press-center" target="_blank" style="text-decoration: none; color: white; margin: 0 15px;">Press Center</a> | 
            <a href="https://www.amazon.in/" target="_blank" style="text-decoration: none; color: white; margin: 0 15px;">Amazon.in</a> | 
            <a href="https://ir.aboutamazon.com/" target="_blank" style="text-decoration: none; color: white; margin: 0 15px;">Investor Resources</a>
       
        <div style="background-color: black; padding: 10px; text-align: center; margin-top: 10px;">
            <a href="https://www.instagram.com/amazon/" target="_blank" style="text-decoration: none;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" alt="Instagram" width="40">
            </a>
            <a href="https://www.facebook.com/amazon" target="_blank" style="text-decoration: none; margin-left: 10px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook" width="40">
            </a>
        </div>
        """

        # Display the footer
        st.markdown(footer_content, unsafe_allow_html=True)
        
    # Recommendations
    with tab2:
        st.subheader("Explore Data")
        
        # Load the data
        data = load_data()

        # Check if data is None before proceeding
        if data is not None and 'user_id' in data.columns:
            le = LabelEncoder()
            data['user_id_encoded'] = le.fit_transform(data['user_id'])
            
            # Create frequency table
            freq_table = pd.DataFrame({
                'User ID': data['user_id_encoded'].value_counts().index,
                'Frequency': data['user_id_encoded'].value_counts().values
            })

            # Pagination settings
            items_per_page = 5
            total_items = len(freq_table)
            total_pages = (total_items // items_per_page) + (total_items % items_per_page > 0)

            # Initialize page state
            if 'page' not in st.session_state:
                st.session_state.page = 0

            # Function to display the current page of data
            def display_page(page):
                start_idx = page * items_per_page
                end_idx = start_idx + items_per_page
                st.table(freq_table.iloc[start_idx:end_idx])

            # Display current page
            display_page(st.session_state.page)

            # Navigation buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Previous"):
                    if st.session_state.page > 0:
                        st.session_state.page -= 1

            with col2:
                if st.button("Next"):
                    if st.session_state.page < total_pages - 1:
                        st.session_state.page += 1

            # Display the current page number
            st.write(f"Page {st.session_state.page + 1} of {total_pages}")

            # Use min and max only if there are user_id_encoded values
            if not data['user_id_encoded'].empty:
                user_id_to_recommend = st.number_input("Enter User ID to get recommendations",
                                                        min_value=data['user_id_encoded'].min(), 
                                                        max_value=data['user_id_encoded'].max())
            else:
                st.warning("No user IDs available for recommendations.")
        else:
            st.warning("Data could not be loaded or missing 'user_id' column.")

        # Recommendation button
        if st.button("Recommend"):
            if 'user_id_to_recommend' in locals():
                recommendations_df = recommend_products(data, user_id_to_recommend)
                st.dataframe(recommendations_df)
            else:
                st.warning("Please select a User ID before requesting recommendations.")

            
   
        
    
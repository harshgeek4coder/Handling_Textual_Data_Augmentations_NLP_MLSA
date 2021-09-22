from os import write
from nlpaug.util import text
from numpy.lib.type_check import mintypecode
import streamlit as st
import nlpaug 
import time
import numpy as np
import pandas as pd
from function_blueprint import char_level_augment,word_level_augment,_tokenizer,sentence_level_augment
import nlpaug.model.word_stats as nmw

st.set_page_config(page_title="Text Augmentation using NLPaug", page_icon=None, layout='centered', initial_sidebar_state='auto')

data_path = 'data\Corona_NLP_train.csv'
df = pd.read_csv(data_path)
df = df.astype(str)

columns_list = list(df.columns)
st.info("QUICK DISCLAIMER : This Application was created only for demonstation of NLPAug Package. I would like to give credit to the main author of this work and highly encourage to check the original work here : https://github.com/makcedward/nlpaug")
st.title("Handling NLP Augmentations using NLPAug Pakcage")
st.write("Dataset Used in this Application can be found here : https://www.kaggle.com/datatattle/covid-19-nlp-text-classification")
st.write(df)


st.subheader("Let's Select and Filter Some Data : ")
selected_column = st.selectbox("Select text data column : ", columns_list)
selected_column = list(df[selected_column])
st.subheader("Now, Let's select a definite Number Of Examples : ")
n_sent = st.slider("Total Number of sentences : ",min_value=0,max_value=100)
text_list = selected_column[:n_sent]
print(n_sent)
print(selected_column[:n_sent])


available_augment_levels = ["Character Level","Word Level","Sentence Level"]
available_char_augmenters = ["KeyboardAug","OcrAug","RandomAug"]
available_char_augmenters_actions = ["substitute", "insert",  "swap", "delete"]

available_word_augmenters = ["AntonymAug","ContextualWordEmbsAug","RandomWordAug","SpellingAug","SplitAug","SynonymAug","TfIdfAug","WordEmbsAug","BackTranslationAug","ReservedAug"]
available_word_augmenters_actions = ["substitute", "insert", "swap", "delete","split","crop"]

available_sentence_augmenters = ["ContextualWordEmbsForSentenceAug","AbstSummAug","LambadaAug"]
available_sentence_augmenters_actions = ["substitute", "insert"]



st.subheader('First, Let us select the augmentation level : \n')
level_option = st.selectbox(
    'Select the Augmentation Level : ',
     available_augment_levels)

if level_option=='Character Level':
    st.subheader("Alright, Let's Select the specific augmenter for {} : ".format(level_option))
    char_augmenter_option = st.selectbox(
        "Select Augmenter : ", 
        available_char_augmenters
    )
    char_augmenter_option_actions = st.selectbox(
        "Select Action : ", available_char_augmenters_actions
    )

    
    st.write(char_augmenter_option,char_augmenter_option_actions)
    try:
        with st.spinner("Please wait.. This might take a while.."):
            text_list_r,augmented_text_r = char_level_augment(char_augmenter_option,text_list,char_augmenter_option_actions)
        st.success("Done..Now Loading Augmentations..")
    
    
    except Exception as e:
        print("ERROR : ",e)
        st.warning("Try another Action..!")
        st.info

    
    st.subheader("Results : ")
    st.subheader("Original Text : ")
    for i in text_list_r:
        st.warning(i)
        
    st.subheader("Augmented Text : ")
    for i in augmented_text_r:
        st.success(i)  
        
    print(augmented_text_r)



elif level_option=="Word Level":
    st.subheader("Alright, Let's Select the specific augmenter for {} : ".format(level_option))
    word_augmenter_option = st.selectbox(
        "Select Augmenter : ", 
        available_word_augmenters
    )
    word_augmenter_option_actions = st.selectbox(
        "Select Action : ", available_word_augmenters_actions
    )

    if word_augmenter_option =="ContextualWordEmbsAug":
        st.warning("NOTE : This Option might take some time to download language models first. Please wait for a while")
        st.info("You can also select a wide variety of different contextual word embeddings like : BERT, DistilBERT, RoBERTa or XLNet language model")
    elif word_augmenter_option =="WordEmbsAug":
        st.warning("NOTE : You have to include the word embedding pre-trained file first in the path. This Option might take some time to download language models first. Please wait for a while")
        st.info("You can also select a wide variety of different contextual word embeddings like :  word2vec, GloVe or fasttext embeddings")

    elif word_augmenter_option =="BackTranslationAug":
        st.warning("NOTE : This Option might take some time to download language models first. Please wait for a while")
           

    
    
    elif word_augmenter_option =="TfIdfAug":
        st.warning("NOTE : This Option might take some time to train TF-IDF Based on your data input. Please wait for a while")
        # Tokenize input
        train_x_tokens = [_tokenizer(x) for x in text_list]

        # Train TF-IDF model
        tfidf_model = nmw.TfIdf()
        tfidf_model.train(train_x_tokens)
        tfidf_model.save('.')
        st.success("TF-IDF trained and file extracted..!")

    else:
        pass

    
    st.write(word_augmenter_option,word_augmenter_option_actions)

    
    try:
        text_list_r,augmented_text_r = word_level_augment(word_augmenter_option,text_list,word_augmenter_option_actions)
    except Exception as e:
        print("ERROR : ",e)
        st.info(e)
        st.warning("Try another Action..!")

    st.subheader("Results : ")
    st.subheader("Original Text : ")
    for i in text_list_r:
        st.warning(i)
        
    st.subheader("Augmented Text : ")
    for i in augmented_text_r:
        st.success(i)  
        
    print(augmented_text_r)


if level_option=='Sentence Level':
    st.subheader("Alright, Let's Select the specific augmenter for {} : ".format(level_option))
    sentence_augmenter_option = st.selectbox(
        "Select Augmenter : ", 
        available_sentence_augmenters
    )
    sentence_augmenter_option_actions = st.selectbox(
        "Select Action : ", available_sentence_augmenters_actions
    )

    
    st.write(sentence_augmenter_option,sentence_augmenter_option_actions)
    try:
        text_list_r,augmented_text_r = sentence_level_augment(sentence_augmenter_option,text_list,sentence_augmenter_option_actions)
    
    
    except Exception as e:
        print("ERROR : ",e)
        st.warning("Try another Action..!")
        st.info

    st.subheader("Results : ")
    st.subheader("Original Text : ")
    for i in text_list_r:
        st.warning(i)
        
    st.subheader("Augmented Text : ")
    for i in augmented_text_r:
        st.success(i)  
        
    print(augmented_text_r)

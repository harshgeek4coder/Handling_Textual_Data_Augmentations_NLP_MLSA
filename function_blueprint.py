import pandas as pd
import numpy as np
import re
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

def _tokenizer(text, token_pattern=r"(?u)\b\w\w+\b"):
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)

def char_level_augment(augmenter_obj,text_list,action):

    if augmenter_obj=='KeyboardAug':
        augmenter = nac.KeyboardAug()
    elif augmenter_obj=='OcrAug':
        augmenter = nac.OcrAug() 
    elif augmenter_obj=='RandomAug':
        augmenter = nac.RandomCharAug(action=action)

    augmented_text = augmenter.augment(text_list)
    print("Original:")
    print(text_list)
    print("Augmented Text:")
    print(augmented_text)

    return text_list,augmented_text


def word_level_augment(augmenter_obj,text_list,action):

    if augmenter_obj=='AntonymAug':
        augmenter = naw.AntonymAug()
    elif augmenter_obj=='ContextualWordEmbsAug':
        augmenter = naw.ContextualWordEmbsAug(action=action) 
    elif augmenter_obj=='RandomWordAug':
        augmenter = naw.RandomWordAug(action=action)
    elif augmenter_obj=='SpellingAug':
        augmenter = naw.SpellingAug() 
    elif augmenter_obj=='SplitAug':
        augmenter = naw.SplitAug()
    elif augmenter_obj=='SynonymAug':
        augmenter = naw.SynonymAug() 
    elif augmenter_obj=='TfIdfAug':
        augmenter = naw.TfIdfAug(action=action,tokenizer=_tokenizer,model_path='.')


    elif augmenter_obj=='WordEmbsAug':
        augmenter = naw.WordEmbsAug(action=action,model_type='word2vec',model_path='.') 
    elif augmenter_obj=='BackTranslationAug':
        augmenter = naw.BackTranslationAug()
    elif augmenter_obj=='ReservedAug':
        augmenter = naw.ReservedAug(action=action)

    augmented_text = augmenter.augment(text_list)
    print("Original:")
    print(text_list)
    print("Augmented Text:")
    print(augmented_text)

    return text_list,augmented_text


def sentence_level_augment(augmenter_obj,text_list,action):

    if augmenter_obj=='ContextualWordEmbsForSentenceAug':
        augmenter = nas.ContextualWordEmbsForSentenceAug()
    elif augmenter_obj=='AbstSummAug':
        augmenter = nas.AbstSummAug() 
    elif augmenter_obj=='LambadaAug':
        augmenter = nas.LambadaAug()

    augmented_text = augmenter.augment(text_list)
    print("Original:")
    print(text_list)
    print("Augmented Text:")
    print(augmented_text)

    return text_list,augmented_text

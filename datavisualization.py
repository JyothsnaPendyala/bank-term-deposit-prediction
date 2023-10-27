from data_preprocessing import data_preprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from extract_data import load_data

def visualise_data():
    df = load_data()
    categorical_features, numerical_features = data_preprocess()
    for categorical_feature in categorical_features:
        fig, axs = plt.subplots(figsize=(5,5))
        sns.countplot(y=categorical_feature,data=df)
        plt.xlabel(categorical_feature)
        plt.title(categorical_feature)
        plt.show()
    for categorical_feature in categorical_features:
        sns.catplot(x='y', col=categorical_feature, kind='count', data=df)
        plt.show()
    for numerical_feature in numerical_features:
        fig, axs = plt.subplots(figsize=(5,4))
        sns.distplot(df[numerical_feature])
        plt.xlabel(numerical_feature)
        plt.show()
    for numerical_feature in numerical_features:
        fig, axs = plt.subplots(figsize=(5,4))
        sns.boxplot(df[numerical_feature])
        plt.xlabel(numerical_feature)
        plt.show()
    cor_mat = df.corr()
    plt.figure(figsize = (12,12))
    sns.heatmap(cor_mat,annot=True)
    plt.show()
    return df

visualise_data()

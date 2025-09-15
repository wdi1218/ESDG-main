# ESDG-main
Domain Adaptation Framework for Multimodal Sentiment Analysis Based on Dynamic Example Selection and Domain Knowledge Generation

# README

## 📖 Project Overview

This repository provides the implementation of our work on **multimodal sentiment analysis with domain adaptation**, leveraging large language models (LLMs) and vision-language models. Experiments are conducted on **MVSA** and **Yelp** datasets, among others. The code covers model downloading, dataset preprocessing, training, and inference.

---

## 📥 Pretrained Models

We employ the following HuggingFace models (recommended to use [hf-mirror](https://hf-mirror.com/) for faster access in certain regions):

1. [Llama-3.1-8B-Instruct](https://hf-mirror.com/meta-llama/Llama-3.1-8B-Instruct)  
   - For source domain dataset fine-tuning and target domain dataset inference

2. [Baichuan2-7B-Base](https://hf-mirror.com/baichuan-inc/Baichuan2-7B-Base)  
   - For source domain dataset fine-tuning and target domain dataset inference

3. [Qwen2.5-VL-7B-Instruct](https://hf-mirror.com/Qwen/Qwen2.5-VL-7B-Instruct)  
   - Used to convert pictures into image captions





## 📊 Datasets

We use three public datasets in our experiments:

1. **MVSA Dataset**
   [MVSA](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)  
   - The dataset includes a variety of topics centered around geographic locations (e.g., Toronto, Montreal, Calgary, Vancouver), events (e.g., concerts, festivals, weddings), lifestyle (e.g., vegan food, fitness, art), and trending social media topics (e.g., popular hashtags, celebrities, fan engagement). The data reflects diverse trends in Canadian culture, entertainment, politics, and local events.

2. **Yelp Dataset** 
   [Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/discussion?sort=undefined)  
   - The dataset centers on consumer reviews of various restaurants and food experiences, highlighting popular cuisines like sushi, pizza, ramen, and vegan options. It captures user sentiments on service, food quality, pricing, and ambiance, with a strong focus on local dining, reservations, and delivery services. Trends reflect both positive and negative feedback on dining experiences, including mentions of specific dishes, locations, and customer service quality.

3. **Twitter2017 Dataset** 
   [Twitter2017](https://www.selectdataset.com/dataset/61d0ffc875f221582d15e56a897cadda)  
   - The dataset covers diverse topics including sports leagues, music concerts, film, fashion, and celebrity culture, alongside politics and social issues. It highlights public attention trends around entertainment events, famous personalities, and global news.

We note that the above descriptions of dataset topics and trends are generated through Domain Knowledge Generation (DKG).


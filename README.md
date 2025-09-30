# ⚡ Spark Yelp Sentiment Analysis  
📊 Sentiment classification of Yelp reviews using **Logistic Regression** implemented in **Apache Spark (PySpark)**.  
The project focuses on **scalability, efficiency, and performance analysis** across different core settings.  

## 📝 Problem Description  
The goal of this project is to classify Yelp reviews into **positive** and **negative** sentiments.  
The challenge comes from handling the **large-scale dataset** (over 6 million reviews) efficiently.  
To solve this, we applied **Logistic Regression** on **Apache Spark**, which allows distributed processing and scalability.  

## 📂 Project Structure  

- **pyspark_app/** → contains all project scripts:  
  - `01_data_ingestion.py` : Load the Yelp dataset.  
  - `02_data_validation.py` : Validate data quality and schema.  
  - `03_data_preprocessing.py` : Text cleaning and preprocessing.  
  - `04_feature_engineering.py` : Feature extraction and transformation.  
  - `05_model_training.py` : Train the Logistic Regression model.  
  - `06_evaluation_metrics.py` : Compute evaluation metrics (Accuracy, Precision, Recall, F1, AUC).  
  - `07_results_analysis.py` : Analyze results and performance.  
  - `08_scaling_plots.py` : Generate scalability and efficiency plots.  

- **runs/** → contains experiment outputs:  
  - **plots/** : Generated plots (duration, efficiency, speedup).  
  - **.csv files** : Reports for training, validation, and scaling experiments.

## 🔄 Main Steps  

1. **Preprocessing**  
   - Cleaning the dataset (removing nulls & duplicates).  
   - Text processing (lowercasing, tokenization, stopword removal).  
   - Feature engineering (N-grams + TF-IDF vectorization).  

2. **Model Training**  
   - Building a Logistic Regression model using **Spark MLlib**.  
   - Training the model on 80% of the data.  

3. **Evaluation**  
   - Testing on 20% of the data.  
   - Metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC.  

4. **Scalability Tests**  
   - Running experiments with different cores (2, 4, 8, 16).  
   - Measuring Execution Time, Efficiency, and Speedup.
  
  ## 📊 Results
- **Accuracy**: ~97%  
- **AUC-ROC**: ~0.99  
- **Scalability**: near-linear speedup up to 8 cores, with diminishing returns after.

## 📊 Plots
**Training**
- Duration vs Cores: <img width="1920" height="1440" alt="training_duration" src="https://github.com/user-attachments/assets/eb791763-2028-45dc-90b6-3320d9ed74bf" />
- Speedup vs Cores: <img width="1920" height="1440" alt="training_speedup" src="https://github.com/user-attachments/assets/c24d0064-0785-4d87-b27d-5ae9ad91c819" />
- Efficiency vs Cores: <img width="1920" height="1440" alt="training_efficiency" src="https://github.com/user-attachments/assets/684ef2cd-8529-4c51-adcf-657997897fbd" />

**Evaluation**
- Duration vs Cores: <img width="1920" height="1440" alt="evaluation_duration" src="https://github.com/user-attachments/assets/5ea6691f-0de3-4d7d-ad4c-776938620c02" />
- Speedup vs Cores: <img width="1920" height="1440" alt="evaluation_speedup" src="https://github.com/user-attachments/assets/f7b82cda-9abd-48f5-a05b-514c618a03a5" />
- Efficiency vs Cores: <img width="1920" height="1440" alt="evaluation_efficiency" src="https://github.com/user-attachments/assets/4f625554-d4db-4a87-9544-99f8cf94351e" />

## ⚙️ Requirements  

To run this project, make sure you have the following installed:  
- **Python 3.8+**  
- **Apache Spark** (with MLlib)  
- **PySpark**  
- **Pandas**  
- **Scikit-learn** (for additional evaluation if needed)  
- **Matplotlib** (for plots)  

## 🔗 Links  
- [📊 Yelp Dataset on Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
- **Code (this repo):** [https://github.com/asmaaalastal/spark-yelp-sentiment](https://github.com/asmaaalastal/spark-yelp-sentiment.git)
## 👩‍💻 Author
**Asmaa Alastal**  
*Supervised by: Dr. Rebhi S. Baraka*




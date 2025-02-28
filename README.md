
# Consumer Complaint Text Classification

This project demonstrates a full pipeline for classifying consumer complaints from the CFPB (Consumer Financial Protection Bureau) dataset into predefined categories. The pipeline includes data exploration, text pre-processing, feature engineering, model training, evaluation, and prediction. This project is implemented in Python using Jupyter Notebook (`customer.ipynb`).

You can download the dataset from [here](https://files.consumerfinance.gov/ccdb/complaints.csv.zip).

## Methods and Algorithms

### 1. Explanatory Data Analysis (EDA) and Feature Engineering
- **Data Inspection:**  
  - Loading the dataset and checking its shape, columns, and missing values.
  - Filtering the dataset to retain relevant columns (`Product` and `Consumer complaint narrative`).
- **Feature Engineering:**  
  - Adding a `text_length` feature which computes the length of each complaint narrative.
  - Mapping the `Product` column to numerical categories based on defined keywords. For example:
    - **Category 0:** Credit reporting, credit card, checking or savings account.
    - **Category 1:** Debt collection.
    - **Category 2:** Payday loan, student loan, vehicle loan.
    - **Category 3:** Mortgage.
- **Visualization:**  
  - Bar plots for category distribution.
  - Histograms for text length distribution.

### 2. Text Pre-Processing
- **Normalization:**  
  - Converting text to lowercase.
- **Cleaning:**  
  - Removing punctuation, numbers, and other non-alphabet characters.
- **Tokenization:**  
  - Splitting text into words using NLTK’s `word_tokenize`.
- **Stop Word Removal:**  
  - Filtering out common English stop words using NLTK’s stopwords corpus.
- **Stemming:**  
  - Reducing words to their base form using NLTK’s `PorterStemmer`.
- **Progress Monitoring:**  
  - Using `tqdm` to show preprocessing progress.

### 3. Model Selection for Multi-Classification
The following models are implemented and compared for classifying complaint narratives:
- **Naive Bayes:**  
  - Utilizes the `MultinomialNB` classifier, which is effective for text classification.
- **Support Vector Machine (SVM):**  
  - Implemented with `LinearSVC` using balanced class weights to handle potential class imbalance.
- **Random Forest:**  
  - An ensemble method using `RandomForestClassifier` with balanced class weights and multiple estimators.

### 4. Comparison of Model Performance
- **Vectorization:**  
  - Transforming pre-processed text into numerical features using TF-IDF vectorization (unigrams and bigrams, with up to 5000 features).
- **Train-Test Split:**  
  - Splitting data into training (80%) and testing (20%) sets using stratification to ensure balanced class distribution.
- **Model Evaluation:**  
  - Each model is trained and evaluated using metrics such as accuracy, precision, recall, and F1-score.
  - Detailed classification reports and confusion matrices are generated for visual inspection.
- **Model Selection:**  
  - The best-performing model is selected based on its accuracy and performance on test data.

### 5. Model Evaluation
- **Confusion Matrix:**  
  - Visual representation of true vs. predicted classifications to identify misclassifications.
- **Classification Report:**  
  - Detailed performance report for each category, providing insights into areas for improvement.

### 6. Prediction
- **Sample Test Case:**  
  - A sample complaint narrative is provided to demonstrate prediction:
    > *"I was unfairly charged on my credit card and my report was negatively affected. I need assistance with credit repair."*
- **Prediction Pipeline:**  
  - The sample text undergoes the same pre-processing and vectorization steps.
  - The selected best model (e.g., Random Forest) predicts the category of the sample text.
  - The predicted category is printed to the console.

## How It Is Useful

- **Practical Application:**  
  - This project provides a real-world example of how to process and classify large volumes of text data, which is applicable in customer service, finance, and complaint management systems.
- **Scalability:**  
  - The pipeline can be extended or integrated into production systems for real-time classification of customer complaints.
- **Insights for Business:**  
  - By categorizing complaints effectively, businesses and regulatory agencies can better understand consumer issues and prioritize resolutions.
- **Educational Value:**  
  - The project serves as a comprehensive case study on text pre-processing, feature engineering, and machine learning model evaluation for multi-class classification problems.

## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Leonallr10/consumer-complaint.git
   cd consumer-complaint
   ```

2. **Set Up a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Unix/Linux
   venv\Scripts\activate      # For Windows
   ```

3. **Install the Required Packages:**
   
   *If you do not have a `requirements.txt` file, manually install:*
   ```bash
   pip install pandas numpy nltk scikit-learn matplotlib seaborn tqdm
   ```

4. **Run the Jupyter Notebook:**
   ```bash
   customer.ipynb
   ```

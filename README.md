# Sentiment Analysis of Financial Reports

## Project Overview
This project automates sentiment analysis of financial reports (10-K filings) to predict stock price trends for S&P 500 companies. The pipeline involves:
- Scraping financial reports from the SEC Edgar database.
- Cleaning and extracting textual data.
- Performing sentiment analysis using the Loughran-McDonald Master Dictionary.
- Correlating sentiment with stock price trends.
- Building predictive models using machine learning.

## Installation
Ensure you have Python installed. Install required dependencies using:
```bash
pip install -r requirements.txt
```

## Pipeline Breakdown

### 1. Scraping Financial Reports
The script `edgar_downloader.py` downloads 10-K filings from the SEC Edgar database for S&P 500 companies.

- Extracts the **CIK Number** for each company.
- Retrieves **10-K filings** from the SEC website.
- Saves HTML reports in a designated folder.

**Usage:**
```python
from edgar_downloader import download_files_10k

download_files_10k("AAPL", "./raw_reports")
```

### 2. Cleaning the Data
The script `edgar_cleaner.py` processes raw HTML files:
- Removes HTML tags and unwanted characters.
- Extracts clean textual content.
- Saves cleaned text as `.txt` files.

**Usage:**
```python
from edgar_cleaner import write_clean_html_text_files

write_clean_html_text_files("./raw_reports", "./clean_reports")
```

### 3. Sentiment Analysis
The script `edgar_sentiment_wordcount.py` categorizes financial reports based on sentiment:
- Uses the **Loughran-McDonald Master Dictionary** to identify positive, negative, and uncertain terms.
- Counts sentiment occurrences per document.
- Outputs results as a CSV file.

**Usage:**
```python
from edgar_sentiment_wordcount import write_document_sentiments

write_document_sentiments("./clean_reports", "./sentiment_data.csv")
```

### 4. Stock Price Data Collection
The script `edgar_reference_data.py` fetches historical stock price data using Yahoo Finance.

**Usage:**
```python
from edgar_reference_data import get_yahoo_data, get_sp100

sp100_tickers = get_sp100()
get_yahoo_data("2013-02-21", "2023-03-08", sp100_tickers)
```

### 5. Machine Learning Models
The following models were trained to predict stock price movements based on sentiment:
- **Linear Regression (`linear_regression.py`)**
- **Decision Tree (`tree_regression.py`)**
- **Random Forest (`random_forest_regression.py`)**

Each model:
- Merges stock price data with sentiment scores.
- Splits data into training and testing sets.
- Trains a model to predict daily stock returns.

Example usage (Decision Tree):
```python
python tree_regression.py
```

## Results and Findings
- The sentiment of financial reports correlates with stock price movements.
- Negative sentiments tend to indicate declining prices.
- Decision Trees and Random Forest models performed better than linear regression.

## Future Improvements
- Integrate deep learning models (LSTMs, Transformers) for better accuracy.
- Expand sentiment analysis to earnings call transcripts.
- Implement real-time updates for financial reports and stock data.

---

**Author:** Nixon Ng  
**GitHub Repository:** [[Nixonnzh]](https://github.com/Nixonnzh)

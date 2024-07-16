from edgar_downloader import download_files_10k
from edgar_cleaner import write_clean_html_text_files
from edgar_reference_data import get_sp100, get_yahoo_data
from edgar_sentiment_wordcount import write_document_sentiments

#tickers = ['AAPL', 'AMZN']

#for ticker in tickers:
    #download_files_10k(ticker, R'C:\edgar\demo_folder_raw')

#write_clean_html_text_files(R'C:\edgar\demo_folder_raw', R'C:\edgar\demo_folder_clean')

#write_document_sentiments(R'C:\edgar\test_folder_clean', R'C:\edgar\test_folder_csv\test_csv.csv')

get_yahoo_data('2013-02-21', '2023-03-08', get_sp100())

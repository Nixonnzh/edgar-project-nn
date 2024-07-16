import os
import pandas as pd
from edgar_reference_data import get_sentiment_word_dict

def write_document_sentiments(input_folder : str, output_file : str):
    '''
    Takes a folder with cleaned html files and outputs a
    dataframe of word sentiments for each year
    '''

    # List of filenames from the input folder
    file_names = os.listdir(input_folder)

    # Define a dictionary to store the sentiment counts
    data=[]
    filenames = []

    # Call dictionary from part 3C
    sent_dict = get_sentiment_word_dict()
    
    

    # Loop through each file in the input folder
    for filename in os.listdir(input_folder):
        
        # Create a dictionary for each file, counting how many different sentiment words are present in that document
        sentiment_counts = {
        'ticker': 0,
        'file_type': 0,
        'filing_date': 0,
        'negative': 0,
        'positive': 0,
        'uncertainty': 0,
        'litigious': 0,
        'constraining': 0,
        'modal': 0,
        'word_count' : 0
        }

        # Read the file
        filenames.append(filename)
        with open(os.path.join(input_folder, filename), 'r') as f:
            text = f.read()

        # Count the number of words in the document belonging to each sentiment
        positive_count = 0
        negative_count = 0
        uncertainty_count = 0
        litigious_count = 0
        constraining_count = 0
        modal_count = 0
        word_count = 0

        for word in text.split():
            if word.upper() in sent_dict['Positive']:
                positive_count += 1
            if word.upper() in sent_dict['Negative']:
                negative_count += 1
            if word.upper() in sent_dict['Uncertainty']:
                uncertainty_count += 1
            if word.upper() in sent_dict['Litigious']:
                litigious_count += 1
            if word.upper() in sent_dict['Constraining']:
                constraining_count += 1
            if word.upper() in sent_dict['Strong_Modal']:
                modal_count += 1
            if word.upper() in sent_dict['Weak_Modal']:
                modal_count += 1
            word_count += 1


        # Add the counts to the sentiment_counts dictionary
        sentiment_counts['positive'] = positive_count
        sentiment_counts['negative'] = negative_count
        sentiment_counts['uncertainty'] = uncertainty_count
        sentiment_counts['litigious'] = litigious_count
        sentiment_counts['constraining'] = constraining_count
        sentiment_counts['modal'] = modal_count
        sentiment_counts['word_count'] = word_count

        # Add the filename chars to dictionary

        file_name = filename.replace('.txt', '').split('_')
        sentiment_counts['ticker'] = file_name[0]
        sentiment_counts['file_type'] = file_name[1]
        sentiment_counts['filing_date'] = file_name[2]

        # Add the dictionary to the list 'data' to get a list of dictionaries, to make an output dataframe
        data.append(sentiment_counts)
        print(f'Added {filename} to csv file')

    with open(output_file, 'w', encoding = 'UTF8', newline = '') as file:
        df = pd.DataFrame.from_dict(data)
        df.to_csv(file, index = False)



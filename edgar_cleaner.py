


import requests
from bs4 import BeautifulSoup
import re
import os
import edgar_downloader




def clean_html_text(html_text : str):
    '''
    help
    '''

    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text()


    clean_text = re.sub('[^A-Za-z0-9]+', ' ', text)

    return clean_text




def write_clean_html_text_files(input_folder : str, dest_folder : str):
    '''
    help message
    '''



    # Get a list of all the file names in the folder
    file_names = os.listdir(input_folder)
    # Loop through the file names and delete each file
    #for file_name in file_names:
    #    file_path = os.path.join(dest_folder, file_name)
    #    os.remove(file_path)

    # Loop through the file names
    for file_name in file_names:
        with open(input_folder +  f'\{file_name}' , 'r') as input_file:
            txt_file_name = file_name.replace('html', 'txt')
            with open(dest_folder + f'\{txt_file_name}', 'w') as output_file:

                html_text = input_file.read()

                clean_file = clean_html_text(html_text)

                output_file.write(clean_file)
                output_file.close()
                print(f'{file_name} cleaned succesfully')






        

        

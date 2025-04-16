import pandas as pd
import os
import random
import re
import numpy as np
import json

#helper functions
def make_dirty(text, strength=0.1):
    """
    This function takes a string and makes it dirty by replacing some characters with their dirty counterparts.
    """
    # Split the text into lines to preserve \n structure
    lines = text.splitlines(keepends=True)
    dirty_lines = []

    for line in lines:
        # replace № with one or two random characters
        replace_variants = ['N', 'N°', 'Nº', 'Ме', 'Не', 'Ле', 'Н', 'Но', 'Ло', 'Мо', '№']
        if random.random() < strength:
            line = re.sub(r'№', random.choice(replace_variants), line)
        
        # replace 0 with one or two random characters
        replace_variants = ['0', 'O', 'o']
        if random.random() < strength:
            line = re.sub(r'0', random.choice(replace_variants), line)
        
        # replace 1 with one or two random characters
        replace_variants = ['1', 'I', 'l', 'i']
        if random.random() < strength:
            line = re.sub(r'1', random.choice(replace_variants), line)
        
        # replace 2 with one or two random characters
        replace_variants = ['2', 'Z', 'z']
        if random.random() < strength:
            line = re.sub(r'2', random.choice(replace_variants), line)

        # replace i with one or two random characters
        replace_variants = ['i', 'I', 'l', '1']
        if random.random() < strength:
            line = re.sub(r'i', random.choice(replace_variants), line)
        
        # replace o with one or two random characters
        replace_variants = ['o', 'O', '0']
        if random.random() < strength:
            line = re.sub(r'o', random.choice(replace_variants), line)

        # replace e with one or two random characters
        replace_variants = ['e', 'E', '3']
        if random.random() < strength:
            line = re.sub(r'e', random.choice(replace_variants), line)
        
        # replace a with one or two random characters
        replace_variants = ['a', 'A', '4']
        if random.random() < strength:
            line = re.sub(r'a', random.choice(replace_variants), line)
        
        # replace s with one or two random characters
        replace_variants = ['s', 'S', '5']
        if random.random() < strength:
            line = re.sub(r's', random.choice(replace_variants), line)
        
        # replace t with one or two random characters
        replace_variants = ['t', 'T', '7']
        if random.random() < strength:
            line = re.sub(r't', random.choice(replace_variants), line)
        
        # replace g with one or two random characters
        replace_variants = ['g', 'G', '9']
        if random.random() < strength:
            line = re.sub(r'g', random.choice(replace_variants), line)
        
        # replace b with one or two random characters
        replace_variants = ['b', 'B', '8']
        if random.random() < strength:
            line = re.sub(r'b', random.choice(replace_variants), line)
        
        # replace l with one or two random characters
        replace_variants = ['l', 'L', '1']
        if random.random() < strength:
            line = re.sub(r'l', random.choice(replace_variants), line)
        
        # randomly change some words by replacing, removing, moving, or adding characters
        words = line.split()
        for i in range(len(words)):
            if random.random() < strength:
                word = words[i]
                if random.random() < 0.5:
                    # remove a random character
                    index = random.randint(0, len(word) - 1)
                    word = word[:index] + word[index + 1:]
                else:
                    # add a random character
                    index = random.randint(0, len(word))
                    char = random.choice('abcdefghijklmnopqrstuvwxyz0123456789 ')
                    word = word[:index] + char + word[index:]
                words[i] = word

        # join the words back together
        line = ' '.join(words)

        # randomly remove some words
        words = line.split()
        for i in range(len(words)):
            if random.random() < strength / 10:
                words[i] = ''
        line = ' '.join(words)

        dirty_lines.append(line)

    # Join the lines back together to preserve \n structure
    return '\n'.join(dirty_lines)


def generate_data_examples(target_folder, result_folder, num_examples=10, min_dirty=0.1, max_dirty=0.5):
    """
    This function generates a number of examples by making the text dirty and saving them to a file.
    """

    dirtiness = np.linspace(min_dirty, max_dirty, num_examples)

    # Create the result folder if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Iterate through all files in the target folder
    for filename in os.listdir(target_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(target_folder, filename)
            
            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                original_text = file.read()
            
            # Create a folder for the current file in the result folder
            file_result_folder = os.path.join(result_folder, os.path.splitext(filename)[0])
            if not os.path.exists(file_result_folder):
                os.makedirs(file_result_folder)
            
            # Generate the specified number of examples
            for i, strength in enumerate(dirtiness):
                # Make the text dirty
                dirty_text = make_dirty(original_text, strength=strength)
                # Save the dirty text to a new file
                output_file_path = os.path.join(file_result_folder, f"example_{i+1}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(dirty_text)
                print(f"Generated {output_file_path} with dirtiness level {strength:.2f}")

def create_json_training_file(json_path, target_folder, examples_folder):
    """
    This function creates a JSON file for training by combining the original and dirty text files.
    """
    # Create a list to hold the data
    data = []

    # Iterate through all files in the target folder
    for filename in os.listdir(target_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(target_folder, filename)
            # Read the content of the original file
            with open(file_path, 'r', encoding='utf-8') as file:
                original_text = file.read()
            
            # Create a folder for the current file in the result folder
            file_result_folder = os.path.join(examples_folder, os.path.splitext(filename)[0])
            
            # Iterate through all dirty examples for the current file
            for example_filename in os.listdir(file_result_folder):
                if example_filename.endswith(".txt"):
                    example_file_path = os.path.join(file_result_folder, example_filename)
                    # Read the content of the dirty example file
                    with open(example_file_path, 'r', encoding='utf-8') as example_file:
                        dirty_text = example_file.read()
                    
                    # Append the data to the list
                    data.append({"text": dirty_text, "target": original_text})
    
    # Write the data to a JSON file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Define the target folder and result folder
    target_folder = 'train-data/target'
    result_folder = 'train-data/examples'
    
    # Generate data examples
    generate_data_examples(target_folder, result_folder, num_examples=10, min_dirty=0.1, max_dirty=0.5)

    # Create the JSON training file
    json_file_path = 'train-data/train_data.json'
    create_json_training_file(json_file_path, target_folder, result_folder)
    print(f"Training data JSON file created at {json_file_path}")
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
    lines = text.splitlines(keepends=True)
    dirty_lines = []

    for line in lines:
        replace_variants = ['N', 'N°', 'Nº', 'Ме', 'Не', 'Ле', 'Н', 'Но', 'Ло', 'Мо', '№']
        if random.random() < strength:
            line = re.sub(r'№', random.choice(replace_variants), line)
        
        # replace 0
        replace_variants = ['0', 'O', 'o']
        if random.random() < strength:
            line = re.sub(r'0', random.choice(replace_variants), line)
        
        # replace 1
        replace_variants = ['1', 'I', 'l', 'i']
        if random.random() < strength:
            line = re.sub(r'1', random.choice(replace_variants), line)
        
        # replace 2
        replace_variants = ['2', 'Z', 'z']
        if random.random() < strength:
            line = re.sub(r'2', random.choice(replace_variants), line)

        # replace i
        replace_variants = ['i', 'I', 'l', '1']
        if random.random() < strength:
            line = re.sub(r'і', random.choice(replace_variants), line)
        
        # replace o
        replace_variants = ['o', 'O', '0']
        if random.random() < strength:
            line = re.sub(r'о', random.choice(replace_variants), line)

        # replace e
        replace_variants = ['e', 'E', '3']
        if random.random() < strength:
            line = re.sub(r'е', random.choice(replace_variants), line)
        
        # replace a
        replace_variants = ['a', 'A', '4']
        if random.random() < strength:
            line = re.sub(r'а', random.choice(replace_variants), line)
        
        # replace з
        replace_variants = ['s', 'S', '5']
        if random.random() < strength:
            line = re.sub(r'з', random.choice(replace_variants), line)
        
        # replace т
        replace_variants = ['t', 'T', '7']
        if random.random() < strength:
            line = re.sub(r'т', random.choice(replace_variants), line)
        
        # replace я
        replace_variants = ['g', 'G', '9']
        if random.random() < strength:
            line = re.sub(r'я', random.choice(replace_variants), line)
        
        # replace в
        replace_variants = ['b', 'B', '8']
        if random.random() < strength:
            line = re.sub(r'в', random.choice(replace_variants), line)
        
        # replace і
        replace_variants = ['l', 'L', '1']
        if random.random() < strength:
            line = re.sub(r'і', random.choice(replace_variants), line)
        
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
                    char = random.choice('abcdefghijklmnopqrstuvwxyz0123456789')
                    word = word[:index] + char + word[index:]
                words[i] = word

        # randomly swap two adjacent words
        for i in range(len(words) - 1):
            if random.random() < strength/10:
                words[i], words[i + 1] = words[i + 1], words[i]
                break

        # randomly split the word into two parts
        for i in range(len(words)):
            if random.random() < strength:
                word = words[i]
                if len(word) < 2:
                    continue
                index = random.randint(1, len(word) - 1)
                words[i] = word[:index] + ' ' + word[index:]
                break
        
        # randomly join two adjacent words
        for i in range(len(words) - 1):
            if random.random() < strength:
                words[i] = words[i] + words[i + 1]
                del words[i + 1]
                break

        # join the words back together
        line = ' '.join(words)

        # randomly remove some words
        words = line.split()
        for i in range(len(words)):
            if random.random() < strength / 10:
                words[i] = ''
        line = ' '.join(words)

        dirty_lines.append(line)

    return '\n'.join(dirty_lines)


def generate_data_examples(target_folder, result_folder, num_examples=10, min_dirty=0.1, max_dirty=0.5):
    dirtiness = np.linspace(min_dirty, max_dirty, num_examples)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for filename in os.listdir(target_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(target_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                original_text = file.read()
            
            file_result_folder = os.path.join(result_folder, os.path.splitext(filename)[0])
            if not os.path.exists(file_result_folder):
                os.makedirs(file_result_folder)

            for i, strength in enumerate(dirtiness):

                dirty_text = make_dirty(original_text, strength=strength)

                output_file_path = os.path.join(file_result_folder, f"example_{i+1}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(dirty_text)
                print(f"Generated {output_file_path} with dirtiness level {strength:.2f}")

def create_json_training_file(json_path, target_folder, examples_folder):
    data = []

    for filename in os.listdir(target_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(target_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                original_text = file.read()
            
            file_result_folder = os.path.join(examples_folder, os.path.splitext(filename)[0])
            
            for example_filename in os.listdir(file_result_folder):
                if example_filename.endswith(".txt"):
                    example_file_path = os.path.join(file_result_folder, example_filename)
                    with open(example_file_path, 'r', encoding='utf-8') as example_file:
                        dirty_text = example_file.read()
                    
                    data.append({"text": dirty_text, "target": original_text})
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def create_json_training_file_per_line(json_path, target_folder, examples_folder):
    data = []

    for filename in os.listdir(target_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(target_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                original_text = file.readlines()
            
            file_result_folder = os.path.join(examples_folder, os.path.splitext(filename)[0])
            
            for example_filename in os.listdir(file_result_folder):
                if example_filename.endswith(".txt"):
                    example_file_path = os.path.join(file_result_folder, example_filename)
                    with open(example_file_path, 'r', encoding='utf-8') as example_file:
                        dirty_text = example_file.readlines()
                    
                    for original_line, dirty_line in zip(original_text, dirty_text):
                        data.append({"text": dirty_line.strip(), "target": original_line.strip()})
    
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    target_folder = 'train-data/target'
    result_folder = 'train-data/examples'
    
    generate_data_examples(target_folder, result_folder, num_examples=15, min_dirty=0.1, max_dirty=0.7)

    json_file_path = 'train-data/train_data.json'
    create_json_training_file(json_file_path, target_folder, result_folder)
    print(f"Training data JSON file created at {json_file_path}")

    json_file_path_per_line = 'train-data/train_data_per_line.json'
    create_json_training_file_per_line(json_file_path_per_line, target_folder, result_folder)
    print(f"Training data JSON file (per line) created at {json_file_path_per_line}")
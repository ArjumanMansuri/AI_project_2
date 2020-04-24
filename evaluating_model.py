import os
import re
import math
from spam_ham_detector_dictionary import *

test_directory = "/Users/zankhanapatel/Documents/git/AI project2/AI_project_2/testdata"


def testModel(test_directory):
    model_file = open("model.txt", "r")
    model_data = []
    for model_line in model_file:
        model_data.append(model_line.strip().split())
    model_file.close()
    result_file = open("result.txt", "w+")
    # to maintain index in result file
    length_counter = 1
    file_list = [os.path.join(test_directory, f) for f in os.listdir(test_directory)]
    for file_path in file_list:
        with open(file_path) as infile:
            # to store type of file 'spam' or 'ham'
            target_result = ''
            if 'spam' in file_path:
                target_result = 'spam'
            elif 'ham' in file_path:
                target_result = 'ham'
            ham_score = math.log10(hamProb)
            spam_score = math.log10(spamProb)
            # Loop through each line of the file
            for line in infile:
                # Remove the leading spaces and newline character
                line = line.strip()
                # Convert the characters in line to lowercase to avoid case mismatch
                lowerLine = str.lower(line)
                validWords = re.split('[^a-zA-Z]', lowerLine)
                # Iterate over each word in line
                for word in validWords:
                    # if the word is not an empty space
                    for data in model_data:
                        if data[1] == word:
                            ham_score += math.log10(float(data[3]))
                            spam_score += math.log10(float((data[5])))
            file_name = file_path.split("/")[-1]
            if ham_score > spam_score:
                actual_result = 'ham'
            else:
                actual_result = 'spam'
            if target_result == actual_result:
                label = 'right'
            else:
                label = 'wrong'
            result_file.write(
                str(length_counter) + '  ' + str(file_name) + '  ' + actual_result + '  ' + str(ham_score) + '  ' + str(
                    spam_score) + '  ' + target_result + '  ' + label + '\n')
            length_counter += 1
    result_file.close()


testModel(test_directory)

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "import math\n",
    "import re\n",
    "import sys\n",
    "import matplotlib.pyplot as plt \n",
    "import os\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data_directory = \"train/\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Word Count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "ham_dictionary = {}    # stores words in ham files and their frequencies\n",
    "spam_dictionary = {}    # stores words in spam files and their frequencies\n",
    "vocabulary = set()    # stores unique words present in all files (spam and ham)\n",
    "\n",
    "def word_count_directory(train_data_directory):\n",
    "    \n",
    "    # list of file paths for files in train_data_directory\n",
    "    file_list = [os.path.join(train_data_directory,f) for f in os.listdir(train_data_directory)]\n",
    "    \n",
    "    # intialize no of spam and ham files\n",
    "    no_of_spam_files = 0\n",
    "    no_of_ham_files = 0\n",
    "    \n",
    "    for file_path in file_list:\n",
    "        with open(file_path,encoding='latin-1') as infile:\n",
    "            # to store type of file 'spam' or 'ham'\n",
    "            file_type = ''\n",
    "            if 'spam' in file_path:\n",
    "                file_type = 'spam'\n",
    "                no_of_spam_files += 1\n",
    "            elif 'ham' in file_path:\n",
    "                file_type = 'ham'\n",
    "                no_of_ham_files += 1\n",
    "                \n",
    "            # Loop through each line of the file \n",
    "            for line in infile:\n",
    "                 \n",
    "                line = line.strip()    # Remove the leading spaces and newline character\n",
    "                lower_line = str.lower(line)    # Convert characters in line to lowercase to avoid case mismatch\n",
    "                valid_words = re.split('[^a-zA-Z]',lower_line) # filter words following the given regex\n",
    "                valid_words = list(filter(None, valid_words))   # filter words with length greater than 0\n",
    "                \n",
    "                # Iterate over each word in line \n",
    "                for word in valid_words:\n",
    "                    if file_type == 'ham':\n",
    "                        # Check if the word is already in dictionary\n",
    "                        if word in ham_dictionary:\n",
    "                            ham_dictionary[word] += 1\n",
    "                        else:\n",
    "                            ham_dictionary[word] = 1     # add word to dictionary with count 1\n",
    "                            vocabulary.add(word)     # add word to vocabulary set\n",
    "\n",
    "                            # if this word is not present in spam_dictionary, add it with count 0\n",
    "                            if word not in spam_dictionary:\n",
    "                                spam_dictionary[word] = 0\n",
    "\n",
    "                    elif file_type == 'spam':\n",
    "                        # Check if the word is already in dictionary\n",
    "                        if word in spam_dictionary:\n",
    "                            spam_dictionary[word] += 1\n",
    "                        else:\n",
    "                            spam_dictionary[word] = 1    # add word to dictionary with count 1\n",
    "                            vocabulary.add(word)    # add word to vocabulary set\n",
    "\n",
    "                            # if this word is not present in ham_dictionary, add it with count 0\n",
    "                            if word not in ham_dictionary:\n",
    "                                ham_dictionary[word] = 0\n",
    "    return no_of_spam_files,no_of_ham_files"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Prior Probabilities"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "no_of_spam_files, no_of_ham_files = word_count_directory(train_data_directory)\n",
    "total_no_of_files = no_of_spam_files + no_of_ham_files\n",
    "prior_prob_of_spam = no_of_spam_files / total_no_of_files\n",
    "prior_prob_of_ham = no_of_ham_files / total_no_of_files"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_model(vocabulary,ham_dictionary,spam_dictionary):\n",
    "    \n",
    "    vocabulary = sorted(vocabulary)    # sorting the vocabulary to maintain order in model.txt\n",
    "    f = open(\"model.txt\",\"w+\")    # creating file that would store the model\n",
    "    N = len(vocabulary)    # getting size of vocabulary\n",
    "    delta = 0.5    # smoothing value\n",
    "    \n",
    "    smoothed_N = (delta * N)\n",
    "    # calculating smoothed denominator for calculating condinational probability of ham words\n",
    "    ham_denominator = sum(ham_dictionary.values()) + smoothed_N\n",
    "    \n",
    "    # calculating smoothed denominator for calculating condinational probability of spam words\n",
    "    spam_denominator = sum(spam_dictionary.values()) + smoothed_N\n",
    "    \n",
    "    for i,word in enumerate(vocabulary):\n",
    "        \n",
    "        freq_in_ham = ham_dictionary[word]    # frequency of word in ham dictionary\n",
    "        c_p_in_ham = (freq_in_ham + delta) / ham_denominator    # conditional probabiltiy of word in ham\n",
    "        freq_in_spam = spam_dictionary[word]    # frequency of word in spam dictionary\n",
    "        c_p_in_spam = (freq_in_spam + delta) / spam_denominator    # conditional probabiltiy of word in spam\n",
    "        \n",
    "        ham_dictionary[word] = c_p_in_ham\n",
    "        spam_dictionary[word] = c_p_in_spam\n",
    "        \n",
    "        # writing all data to model.txt\n",
    "        f.write(str(i+1)+'  '+word+'  '+str(freq_in_ham)+'  '+str( \"{:.8f}\".format(float( c_p_in_ham )) )+'  '+str(freq_in_spam)+'  '+str( \"{:.8f}\".format(float( c_p_in_spam )) )+'\\n')\n",
    "    \n",
    "    f.close()    # closing the file\n",
    "    \n",
    "create_model(vocabulary,ham_dictionary,spam_dictionary)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluate the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data_directory = \"test/\"\n",
    "\n",
    "# applying log10 on prior probabilities\n",
    "log_of_ham = math.log10(prior_prob_of_ham)\n",
    "log_of_spam = math.log10(prior_prob_of_spam)\n",
    "\n",
    "# initialising variables needed for confusion matrix\n",
    "true_positive = 0   # correct Ham -> result Ham\n",
    "true_negative = 0    # correct Spam -> result Spam\n",
    "false_positive = 0    # correct Spam -> result Ham\n",
    "false_negative = 0    # correct Ham -> result Spam\n",
    "\n",
    "file_list = [os.path.join(test_data_directory,f) for f in os.listdir(test_data_directory)]    # file paths of test files\n",
    "temp_counter = 0    # counter to store the test file count\n",
    "\n",
    "f = open(\"result.txt\", \"w+\")   # 'w+' for reading and writing\n",
    "f.truncate(0)\n",
    "\n",
    "for file_path in file_list:\n",
    "    \n",
    "    with open(file_path,encoding = 'latin-1') as infile:\n",
    "\n",
    "        file_name = file_path.rsplit('/',1)[1]    # file name to store in result.txt\n",
    "        temp_counter = temp_counter + 1\n",
    "        score_log_ham = log_of_ham     # score for ham\n",
    "        score_log_spam = log_of_spam    # score for spam\n",
    "\n",
    "        if(\"test-ham\" in file_path):\n",
    "            correct_classification = \"ham\"\n",
    "        else:\n",
    "            correct_classification = \"spam\"\n",
    "\n",
    "        vocab_test = []    #  stores words in test file\n",
    "        for line in infile:\n",
    "\n",
    "            line = line.strip()    # Remove the leading spaces and newline character\n",
    "            lower_line = str.lower(line)    # Convert characters in line to lowercase to avoid case mismatch\n",
    "            valid_words = re.split('[^a-zA-Z]',lower_line) # filter words following the given regex\n",
    "            valid_words = list(filter(None, valid_words))   # filter words with length greater than 0\n",
    "            vocab_test = vocab_test + valid_words    # appending valid_words to vocab_test\n",
    "\n",
    "\n",
    "        for word in vocab_test:\n",
    "            if word in vocabulary:\n",
    "                # add log10 of conditional probability of word in ham_dictionary\n",
    "                score_log_ham = score_log_ham + math.log10(ham_dictionary[word])\n",
    "                # add log10 of conditional probability of word in ham_dictionary\n",
    "                score_log_spam = score_log_spam + math.log10(spam_dictionary[word])\n",
    "        \n",
    "\n",
    "        if(score_log_ham > score_log_spam):\n",
    "            predicted_classification = \"ham\"\n",
    "        else:\n",
    "            predicted_classification = \"spam\"\n",
    "\n",
    "        if(correct_classification == predicted_classification):\n",
    "            label = \"right\"\n",
    "        else:\n",
    "            label = \"wrong\"\n",
    "\n",
    "        if(correct_classification == \"ham\" and predicted_classification == \"ham\"):\n",
    "            true_positive = true_positive + 1\n",
    "            \n",
    "        elif(correct_classification == \"spam\" and predicted_classification == \"spam\"):\n",
    "            true_negative = true_negative + 1\n",
    "            \n",
    "        elif(correct_classification == \"spam\" and predicted_classification == \"ham\"):\n",
    "            false_positive = false_positive + 1\n",
    "            \n",
    "        elif(correct_classification == \"ham\" and predicted_classification == \"spam\"):\n",
    "            false_negative = false_negative + 1\n",
    "\n",
    "        # format scores to appropriate string value\n",
    "        score_log_ham = str( \"{:.8f}\".format(float(score_log_ham)))\n",
    "        score_log_spam = str( \"{:.8f}\".format(float(score_log_spam)))\n",
    "        \n",
    "        # writing results to result.txt\n",
    "        f.write(str(str(temp_counter)+\" \"+str(file_name)+\" \"+str(predicted_classification)+\" \"+str(score_log_ham)+\" \"+str(score_log_spam)+\" \"+str(correct_classification)+\" \"+str(label)+\"\\n\"))\n",
    "\n",
    "f.close()    # closing file\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def confusion_matrix(values , class_name , title='Confusion matrix'):\n",
    "\n",
    "    label_array=np.array([[ \"(True Positive)\",  \"(False Positive)\"],\n",
    "                                              [  \"(False Negative)\",  \"(True Negative)\"]])\n",
    "   \n",
    "    plt.figure(figsize=(4, 4))\n",
    "    plt.imshow(values)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick = np.arange(len(class_name))\n",
    "    plt.xticks(tick, class_name)\n",
    "    plt.yticks(tick, class_name)\n",
    "\n",
    "\n",
    "    for i in range (values.shape[0]):\n",
    "        for j in range (values.shape[1]):    \n",
    "            plt.text(j, i, \"{:,}\".format(values[i, j]),\n",
    "            color=\"red\",horizontalalignment=\"center\",verticalalignment=\"bottom\")\n",
    "            plt.text(j, i,label_array[i,j],\n",
    "            color=\"red\",horizontalalignment=\"center\",verticalalignment=\"top\")\n",
    "\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('Predicted ')\n",
    "    plt.xlabel('Actual ')     \n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 288x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "confusion_matrix(values= np.array([[ true_positive, false_positive ],\n",
    "                                    [  false_negative,  true_negative]]), class_name = ['Ham', 'Spam'],title=\"Confusion matrix for Ham Class\")\n",
    "\n",
    "\n",
    "confusion_matrix(values= np.array([[ true_negative , false_negative ],\n",
    "                                    [  false_positive,  true_positive]]), class_name = ['Spam', 'Ham'],title=\"Confusion matrix for Spam class\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Evaluation Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_ham = true_positive/(true_positive+false_negative)*100\n",
    "accuracy_spam = true_negative/(true_negative+false_positive)*100\n",
    "\n",
    "accuracy_model=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)*100\n",
    "\n",
    "precision_ham = true_positive/(true_positive+false_positive)\n",
    "precision_spam =  true_negative/(true_negative+false_negative)\n",
    "\n",
    "recall_ham =   true_positive/ (true_positive + false_negative)\n",
    "recall_spam =  true_negative/(true_negative+false_positive)\n",
    "\n",
    "F1measure_ham =  2*(recall_ham * precision_ham) / (recall_ham + precision_ham)\n",
    "F1measure_spam = 2*(recall_spam * precision_spam) / (recall_spam + precision_spam)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAqwAAADnCAYAAADbwgS2AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO3deVxU9f4/8NcgKuJCbnVZVECQZZgZwA0KFSQR1yvulOZC5b25XDW3e820m14rLbU0b19zzx92NRVLb1qiiNtVFNywUGLXVFABQZBh3r8/uJyvEwNqmhy/vZ6Phw+Zs3zm8znzOZ95zZlzzmhEBEREREREamVV2xUgIiIiIqoJAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREamadU0zGzRo8HNJSclzT6oy9HSwsbExlZSU8MMOmWG/IEvYL8gS9guyxMbG5uqdO3f+YGmepqYfDtBoNMIfFqBf0mg0YL+gX2K/IEvYL8gS9guy5L/9QmNpHj/dEBEREZGqMbASERERkaoxsBIRERGRqjGwEhEREZGqMbASERERkaoxsBIRERGRqv1uA+vPP/+M4cOHo23btvD29kbv3r2RkpICHx+f2q4a/Ua2b98OjUaDH374obarQipRp04d+Pr6wsfHB0OGDEFxcfEjl5mQkIBJkyZVO//y5csYPHjwIz8P1Z57+02/fv1w69atx1r+unXrMGHCBADAvHnzsHjx4sdaPlWv8rWt/Jeeno68vDyEhISgUaNGyutCT97vMrCKCCIiIhAcHIzU1FQkJyfjH//4B65evVrbVaPfUHR0NIKCgrB58+bf7DnKy8t/s7Lp8WvQoAGSkpJw7tw51KtXD//85z/N5osITCbTQ5XZoUMHfPzxx9XOd3BwwNatW39VfUkd7u03zZo1w4oVK2q7SvSYVL62lf+cnZ1hY2ODd99996n84PB/6T3pdxlY9+/fj7p16+JPf/qTMs3X1xetWrVSHqenp6NLly7w9/eHv78/jhw5AgC4cuUKunbtqny6jo+PR3l5OUaPHg0fHx/odDosWbLkibeJanb79m0cPnwYq1evNgusH3zwAXQ6HQwGA2bNmgUAuHTpEl588UUYDAb4+/sjNTUVBw4cQN++fZX1JkyYgHXr1gEAnJ2d8fe//x1BQUHYsmULVq1ahY4dO8JgMGDQoEHKUburV68iIiICBoMBBoMBR44cwZw5c7Bs2TKl3NmzZ9cYdui306VLF1y6dAnp6enw8vLCG2+8AX9/f2RlZWHv3r0IDAyEv78/hgwZgtu3bwMATpw4geeffx4GgwGdOnVCYWGhWV+Ji4tTjtT4+fmhsLAQ6enpyjc5JSUlGDNmDHQ6Hfz8/LB//34AFUfYBg4ciPDwcLi7u2PGjBm1s1HovgIDA5GTk6M8XrRoETp27Ai9Xo+5c+cq0zds2AC9Xg+DwYCRI0cCAL7++mt07twZfn5+ePHFF3nQRKUaNmyIoKAg2NjY1LjcunXrMGDAAPTr1w8uLi5Yvnw5PvroI/j5+SEgIAA3btwAAKSmpiI8PBzt27dHly5dlG/9qusPlsaRh3lPqu75njoiUu2/itn/9yxbtkwmT55cZXpaWppotVoRESkqKpI7d+6IiEhKSoq0b99eREQWL14s8+fPFxERo9EoBQUFkpCQIC+++KJSzs2bN3/rJtSqp7FfbNy4UcaOHSsiIoGBgXLy5EnZvXu3BAYGSlFRkYiI5OXliYhIp06dZNu2bSIicufOHSkqKpL9+/dLnz59lPLGjx8va9euFRGRNm3ayPvvv6/My83NVf6ePXu2fPzxxyIiMnToUFmyZImIVPSdW7duSVpamvj5+YmISHl5ubi6upqt/zR5GvtFw4YNRUSkrKxM+vfvL59++qmkpaWJRqORo0ePiojI9evXpUuXLnL79m0REXnvvffknXfekdLSUnFxcZHjx4+LiEh+fr6UlZWZ9ZW+ffvKoUOHRESksLBQysrKzMaZxYsXy+jRo0VE5MKFC9KqVSu5c+eOrF27VlxcXOTWrVty584dad26tWRmZj65DfMYPY394n4q+43RaJTBgwfLv//9bxER2bNnj7z22mtiMpmkvLxc+vTpI3FxcXLu3Dlp166dXL9+XUT+d6y5ceOGmEwmERFZtWqVTJ06VURE1q5dK+PHjxcRkblz58qiRYueaPueBLX2CysrKzEYDGIwGGTAgAFm8+59XSxZu3attG3bVgoKCuTatWvSpEkTWblypYiITJ48WRn/u3fvLikpKSIicuzYMQkJCRGR6vuDpXHkYd6Tqns+Nfpvv7CYSa1rMSurWllZGSZMmICkpCTUqVMHKSkpAICOHTti7NixKCsrw4ABA+Dr6wtXV1f89NNPmDhxIvr06YOwsLBarj39UnR0NCZPngwAGD58OKKjo2EymTBmzBjY2toCAJo1a4bCwkLk5OQgIiICAO77ibrSsGHDlL/PnTuHt956C7du3cLt27fRs2dPAEBsbCw2bNgAoOI8KTs7O9jZ2aF58+ZITEzE1atX4efnh+bNmz+2dlPN7ty5A19fXwAVR1ijoqJw+fJltGnTBgEBAQCAY8eOITk5GS+88AIA4O7duwgMDMSPP/4Ie3t7dOzYEQDQpEmTKuW/8MILmDp1Kl5++WUMHDgQTk5OZvMPHTqEiRMnAgA8PT3Rpk0bZawJDQ2FnZ0dAMDb2xsZGRlm3wJR7ansN+np6Wjfvj169OgBANi7dy/27t0LPz8/ABXf7Fy8eBGnT5/G4MGD0aJFCwAVYw0AZGdnY9iwYbhy5Qru3r0LFxeX2mkQKSpPCfi1QkJC0LhxYzRu3Bh2dnbo168fAECn0+HMmTO4ffs2jhw5giFDhijrlJaWAqi+P9xvHLGk8j2ppud72vwuTwnQarU4efJkjcssWbIEzz33HE6fPo2EhATcvXsXANC1a1ccPHgQjo6OGDlyJDZs2ICmTZvi9OnTCA4OxooVK/Dqq68+iWbQA8rLy0NsbCxeffVVODs7Y9GiRfjyyy9hMpmg0Zj/ZLFU89vW1tbWZucylpSUmM1v2LCh8vfo0aOxfPlynD17FnPnzq2y7C+9+uqrWLduHdauXYuxY8c+bPPoEdx7vtonn3yCevXqATB/PUUEPXr0UJZLTk7G6tWrISJV+s8vzZo1C59//jnu3LmDgICAKl/FVdffAKB+/frK33Xq1IHRaPw1TaTfQGW/ycjIwN27d5VzWEUEf/3rX5W+cunSJURFRVXbVyZOnIgJEybg7Nmz+Oyzz+47VpC6bN++XfmqPiEhAYD5fmtlZaU8trKygtFohMlkwjPPPGN2nuyFCxcAVN8fLI0jD/qeVNPzPW1+l4G1e/fuKC0txapVq5RpJ06cQEZGhvI4Pz8f9vb2sLKywsaNG5UTlzMyMvDss8/itddeQ1RUFE6dOoXc3FyYTCYMGjQI7777Lk6dOvXE20TV27p1K1555RVkZGQgPT0dWVlZcHFxQbNmzbBmzRrlHNMbN26gSZMmcHJywo4dOwBUfBItLi5GmzZtkJycjNLSUuTn52Pfvn3VPl9hYSHs7e1RVlaGTZs2KdNDQ0OxcuVKABUnwhcUFAAAIiIi8O233+LEiRPK0VhSj4CAABw+fBiXLl0CABQXFyMlJQWenp64fPkyTpw4AaDidf9lqExNTYVOp8PMmTPRoUOHKoG1a9euSh9JSUlBZmYmPDw8nkCr6HGws7PDxx9/jMWLF6OsrAw9e/bEmjVrlHOcc3JycO3aNYSGhuJf//oX8vLyAEA5lzE/Px+Ojo4AgPXr19dOI+hXi4iIUEJghw4dHmidJk2awMXFBVu2bAFQ8SHn9OnTAKrvD5bGkQd9T6rp+Z42v8vAqtFosH37dnz33Xdo27YttFot5s2bBwcHB2WZN954A+vXr0dAQABSUlKUTysHDhxQTnz+6quv8Je//AU5OTkIDg6Gr68vRo8ejYULF9ZW08iC6Oho5Sv+SoMGDcLly5fRv39/dOjQAb6+vsoVoBs3bsTHH38MvV6P559/Hj///DNatWqFoUOHQq/X4+WXX1a+8rPk3XffRefOndGjRw94enoq05ctW4b9+/dDp9Ohffv2OH/+PACgXr16CAkJwdChQ1GnTp3fYAvQo2jZsiXWrVuHyMhI6PV65QhHvXr18OWXX2LixIkwGAzo0aNHlaMcS5cuhY+PDwwGAxo0aIBevXqZzX/jjTdQXl4OnU6HYcOGYd26dWZHaEj9/Pz8YDAYsHnzZoSFheGll15CYGAgdDodBg8ejMLCQmi1WsyePRvdunWDwWDA1KlTAVTcsmrIkCHo0qWLcroAqZOzszOmTp2KdevWwcnJCcnJyb+6rE2bNmH16tUwGAzQarWIiYkBUH1/sDSOPMx7UnXP97TR1PSVlEajkZrm0++TRqOp8atMejgmkwn+/v7YsmUL3N3da7s6vxr7BVnCfkGWsF+QJf/tFxbPtfpdHmElUovk5GS4ubkhNDT0qQ6rREREvyUeYaWHxk/GZAn7BVnCfkGWsF+QJTzCSkRERERPLQZWIiIiIlI1BlYiIiIiUjUGViIiIiJSNQZWIiIiIlI165pm2tjYmDQaDUMtmbGxsbnvT1LS7w/7BVnCfkGWsF+QJTY2Nqbq5vG2VvTQeDsSsoT9gixhvyBL2C/IEt7WioiIiIieWgysRERERKRqDKxEREREpGoMrERERESkagysRERERKRqDKxEREREpGoMrE+BRo0amT1et24dJkyYUEu1IbVZtmwZfHx8oNVqsXTpUgBAUlISAgIC4Ovriw4dOuD48eMW161Tpw58fX3h6+uL/v37P8lq49tvv4WHhwfc3Nzw3nvvVZmfmZmJkJAQ+Pn5Qa/XY/fu3cq8M2fOIDAwEFqtFjqdDiUlJQCAkydPQqfTwc3NDZMmTVJumzN9+nR4enpCr9cjIiICt27dqrGs4uJi9OnTB56entBqtZg1a5ayfGlpKYYNGwY3Nzd07twZ6enpyryFCxfCzc0NHh4e2LNnjzJ9yZIl0Gq18PHxQWRkpFLfLl26KNvfwcEBAwYMAADExMRAr9crr9+hQ4cAABkZGWjfvj18fX2h1Wrxz3/+U3mOu3fv4vXXX0e7du3g6emJr776CgDw0UcfwdvbG3q9HqGhocjIyDDbzgUFBXB0dDQbU4KDg+Hh4aHU7dq1aw/ykpKK3G//ysjIQGhoKPR6PYKDg5Gdna3MmzFjBrRaLby8vMz2o9rqF4/SlvDwcDzzzDPo27ev2Tr79u2Dv78/fH19ERQUhEuXLpnN37p1KzQaDRISEgAAeXl5CAkJQaNGjap9/+3fvz98fHyUx6dPn0ZgYCB0Oh369euHgoKC+5YVHR0NnU4HvV6P8PBw5Obmms1fvHgxNBqNMv2HH35AYGAg6tevj8WLF1epU3l5Ofz8/MzaHxsbC39/f/j4+GDUqFEwGo33Lau6MWz06NFwcXFR+kRSUpLFbfNYiEi1/ypmU21r2LCh2eO1a9fK+PHja6k2IuwX6nH27FnRarVSVFQkZWVlEhoaKikpKdKjRw/ZvXu3iIjs2rVLunXrZnH9X/atR/Ew/cJoNIqrq6ukpqZKaWmp6PV6OX/+vNkyr732mnz66aciInL+/Hlp06aNiIiUlZWJTqeTpKQkERHJzc0Vo9EoIiIdO3aUI0eOiMlkkvDwcGUb7NmzR8rKykREZMaMGTJjxowayyoqKpLY2FgRESktLZWgoCClrBUrVsi4ceNERCQ6OlqGDh2q1FGv10tJSYn89NNP4urqKkajUbKzs8XZ2VmKi4tFRGTIkCGydu3aKttk4MCBsn79ehERKSwsFJPJJCIip0+fFg8PD6UuJSUlyjJt2rSRnJwcERF5++23Zfbs2SIiUl5eLtevXxcRkdjYWCkqKhIRkU8//VSpb6VJkyZJZGSk2ZjSrVs3OXHihMXX7mFxvHjyHmT/Gjx4sKxbt05ERPbt2ycjRowQEZHDhw/L888/L0ajUYxGowQEBMj+/ftFpHb6xaO0RUTk+++/l507d0qfPn3M1nF3d5fk5GQRqdinR40apcwrKCiQLl26SOfOnZX23r59W+Lj42XlypUW33+/+uoriYyMFK1Wq0zr0KGDHDhwQEREVq9eLW+99VaNZZWVlUnLli2VfXf69Okyd+5cZX5mZqaEhYVJ69atlWWuXr0qx48fl7/97W+yaNGiKvX68MMPJTIyUml/eXm5ODk5yY8//igiInPmzJHPP/+8xrJqGsNGjRolW7ZsqfK8v9Z/+4XFTMojrE+5r7/+Gp07d4afnx9efPFFXL16FQAwb948jBo1CmFhYXB2dsa2bdswY8YM6HQ6hIeHo6ysrJZrTo/DhQsXEBAQAFtbW1hbW6Nbt27Yvn07NBqN8mk+Pz8fDg4OtVxTc8ePH4ebmxtcXV1Rr149DB8+HDExMWbLVNeGvXv3Qq/Xw2AwAACaN2+OOnXq4MqVKygoKEBgYCA0Gg1eeeUV7NixAwAQFhYGa+uKH/YLCAhQjsBUV5atrS1CQkIAAPXq1YO/v7+yTkxMDEaNGgUAGDx4MPbt2wcRQUxMDIYPH4769evDxcUFbm5uypFto9GIO3fuwGg0ori4uMrrUVhYiNjYWOUIa6NGjZRfASoqKlL+rlevHurXrw+g4kivyfS/PwqzZs0a/PWvfwUAWFlZoUWLFgCAkJAQ2NraVmk7UHFE+urVqwgLC3vQl46eAg+yfyUnJyM0NBRARR+pnK/RaFBSUoK7d++itLQUZWVleO655554Gyo9SlsAIDQ0FI0bN65Sbk1j5Jw5czBjxgzY2Ngo0xo2bIigoCCzaZVu376Njz76CG+99ZbZ9B9//BFdu3YFAPTo0UP51qO6siqDWVFREUQEBQUFZvWaMmUKPvjgA7NfCHv22WfRsWNH1K1bt0q9srOzsWvXLrz66qvKtLy8PNSvXx/t2rWrUq+ayrrfGPYkMLA+Be7cuaMcbvf19cXbb7+tzAsKCsKxY8eQmJiI4cOH44MPPlDmpaamYteuXYiJicGIESMQEhKCs2fPokGDBti1a1dtNIUeMx8fHxw8eBB5eXkoLi7G7t27kZWVhaVLl2L69Olo1aoVpk2bhoULF1pcv6SkBB06dEBAQIAS7p6EnJwctGrVSnns5OSEnJwcs2XmzZuHL774Ak5OTujduzc++eQTAEBKSgo0Gg169uwJf39/pc/n5OTAycmpxjKBimDXq1evGsu6161bt/D1118rb4j31t3a2hp2dnbIy8urtk2Ojo6YNm0aWrduDXt7e9jZ2VUJiNu3b0doaCiaNGliNs3T0xN9+vTBmjVrlOlZWVnQ6/Vo1aoVZs6cCQcHB+UUhzlz5sDf3x9DhgxRPrzea/Xq1UrbTSYT3nzzTSxatKjKcgAwZswY+Pr64t133+UvEj1lHmT/MhgMSlDZvn07CgsLkZeXh8DAQISEhMDe3h729vbo2bMnvLy8lPWedL94lLbU5PPPP0fv3r3h5OSEjRs3Kqf9JCYmIisrq8opBDWZM2cO3nzzTeWDYSUfHx/s3LkTALBlyxZkZWXVWE7dunWxcuVK6HQ6ODg4IDk5GVFRUQCAnTt3wtHRUflw/SAmT56MDz74AFZW/xv1WrRogbKyMuVUh61bt963Xvcbw2bPng29Xo8pU6agtLT0gev3sBhYnwINGjRAUlKS8u/vf/+7Mi87Oxs9e/aETqfDokWLcP78eWVer169ULduXeh0OpSXlyM8PBwAoNPpzM67o6eXl5cXZs6ciR49eiA8PBwGgwHW1tZYuXIllixZgqysLCxZskQZ9H4pMzMTCQkJ+H//7/9h8uTJSE1NfSL1tvRG98vfFY+Ojsbo0aORnZ2N3bt3Y+TIkTCZTDAajTh06BA2bdqEQ4cOYfv27cpRzvuVuWDBAlhbW+Pll18GgGrLqmQ0GhEZGYlJkybB1dW1xrpXN/3mzZuIiYlBWloaLl++jKKiInzxxRdV2hoZGWk2LSIiAj/88AN27NiBOXPmKNNbtWqFM2fO4NKlS1i/fj2uXr0Ko9GI7OxsvPDCCzh16hQCAwMxbdo0s/K++OILJCQkYPr06QCATz/9FL179zYLA5U2bdqEs2fPIj4+HvHx8di4cWOVZUi9HmRfWLx4MeLi4uDn54e4uDg4OjrC2toaly5dwoULF5CdnY2cnBzExsbi4MGDAGqnXzxKW2qyZMkS7N69G9nZ2RgzZgymTp0Kk8mEKVOm4MMPP3zg+iUlJeHSpUuIiIioMm/NmjVYsWIF2rdvj8LCQtSrV6/GssrKyrBy5UokJibi8uXL0Ov1WLhwIYqLi7FgwQKz9/77+eabb/Dss8+iffv2ZtM1Gg02b96MKVOmoFOnTmjcuPF9t1VNY9jChQvxww8/4MSJE7hx4wbef//9B67jw2JgfcpNnDgREyZMwNmzZ/HZZ58pJ0IDUL46tLKyQt26dZWd3MrKSjnJmp5+UVFROHXqFA4ePIhmzZrB3d0d69evx8CBAwEAQ4YMqfaiq8qvdVxdXREcHIzExMQnUmcnJyezT/XZ2dlVvmJavXo1hg4dCgAIDAxESUkJcnNz4eTkhG7duqFFixawtbVF7969cerUKTg5OZl93f3LMtevX49vvvkGmzZtUvaF6sqq9Prrr8Pd3R2TJ0+2WHej0Yj8/Hw0a9as2jZ9//33cHFxQcuWLVG3bl0MHDgQR44cUZbLy8vD8ePH0adPH4vbqmvXrkhNTa1y8YWDgwO0Wi3i4+PRvHlz2NraKm+aQ4YMMWvH999/jwULFmDnzp3KuHD06FEsX74czs7OmDZtGjZs2KAcZXJ0dAQANG7cGC+99FK1/YfU6UH2LwcHB2zbtg2JiYlYsGABAMDOzg7bt29HQEAAGjVqhEaNGqFXr144duwYgNrpF4/Slupcv34dp0+fRufOnQEAw4YNw5EjR1BYWIhz584hODgYzs7OOHbsGPr3768cjbTk6NGjOHnyJJydnREUFISUlBQEBwcDADw9PbF3716cPHkSkZGRaNu2bY1trbxgqW3bttBoNBg6dCiOHDmC1NRUpKWlwWAwwNnZGdnZ2fD398fPP/9cbVmHDx/Gzp074ezsjOHDhyM2NhYjRowAUDGexsfH4/jx4+jatSvc3d1rrFdNY5i9vT00Gg3q16+PMWPG/KZ9goH1KZefn68MIuvXr6/l2lBtqLxSNzMzE9u2bUNkZCQcHBwQFxcHoOKKUEsD0s2bN5Wvb3Jzc3H48GF4e3s/kTp37NgRFy9eRFpaGu7evYvNmzdXuUtB69atlaOdFy5cQElJCVq2bImePXvizJkzKC4uhtFoRFxcHLy9vWFvb4/GjRvj2LFjEBFs2LABf/zjHwFUXGX8/vvvY+fOnWZf21VXFgC89dZbyM/PV+68UKl///7KvrZ161Z0794dGo0G/fv3x+bNm1FaWoq0tDRcvHgRnTp1QuvWrXHs2DEUFxdDRLBv3z6zr1i3bNmCvn37mp3PdunSJeXI0qlTp3D37l00b94c2dnZuHPnDoCK1+/w4cPw8PCARqNBv379cODAAQAVV0BXtiMxMRHjxo3Dzp078eyzzyrPsWnTJmRmZiI9PR2LFy/GK6+8gvfeew9Go1EJx2VlZfjmm2/Mrnwm9XuQ/Ss3N1c5B3rhwoUYO3YsgIr9Li4uDkajEWVlZYiLi4OXl1et9YtHaUt1mjZtivz8fKSkpAAAvvvuO3h5ecHOzg65ublIT09Heno6AgICsHPnTnTo0KHasv785z/j8uXLSE9Px6FDh9CuXTtlP6wcm00mE+bPn48//elPNdbL0dERycnJuH79ulm9dDodrl27ptTLyckJp06dwh/+8Idqy1q4cCGys7ORnp6OzZs3o3v37spR0cp6lZaW4v33379vvWoaw65cuQKg4kj4jh07fts+Ud3VWMK7BKhGTXcJ2LFjh7i4uEhQUJBMmzZNuRp87ty5Zlf53VvGL+c9LPYLdQkKChIvLy/R6/Xy/fffi4hIfHy8+Pv7i16vl06dOklCQoKIiJw4cUKioqJEpOJqYB8fH9Hr9eLj46NcKfprPWy/2LVrl7i7u4urq6vMnz9fRCquWI2JiRGRiqvun3/+edHr9WIwGGTPnj3Kuhs3bhRvb2/RarUyffp0ZfqJEydEq9WKq6urjB8/XrnSvm3btuLk5CQGg0EMBoNylX91ZWVlZQkA8fT0VNZZtWqViIjcuXNHBg8eLG3btpWOHTtKamqqUtb8+fPF1dVV2rVrp9xVQKTiCn4PDw/RarUyYsQI5Up/kYorr//973+bbZv33ntPvL29xWAwSEBAgMTHx4uIyN69e0Wn04lerxedTiefffaZsk56erp06dJFdDqddO/eXTIyMkREJDQ0VJ599lmlHf369avyWtw7pty+fVv8/f1Fp9OJt7e3TJo0SbkLw6/B8aJ23G//2rJli7i5uYm7u7tERUUpfdJoNMrrr78unp6e4uXlJVOmTBGR2u0Xv7YtIhXjY4sWLcTGxkYcHR3l22+/FRGRbdu2KeNft27dzPbjSr+8K0KbNm2kadOm0rBhQ3F0dKxyt4K0tDSzuwQsXbpU3N3dxd3dXWbOnKmMRzWVtXLlSvH09BSdTid9+/aV3NzcKvVq06aNcpeAK1euiKOjozRu3Fjs7OzE0dFR8vPzzZbfv3+/2V0Spk2bJp6entKuXTtZsmSJMr2msqobw0JCQsTHx0e0Wq28/PLLUlhYWKW+DwM13CVAIzWcNK3RaKSm+fT7VN35evT7xn5BlrBfkCXsF2TJf/uFxtI8nhJARERERKrGwEpEREREqsbASkRERESqxsBKRERERKrGwEpEREREqsbASkRERESqxsBKRERERKpW4w/I2tjYmDQaDUMtmbGxsanyW85E7BdkCfsFWcJ+QZbY2NiYqpvHHw6gh8YbPpMl7BdkCfsFWcJ+QZbwhwOIiIiI6KnFwEpEREREqsbASkRERESqxsBKRERERKrGwEpEREREqsbASkRERESqxsCqQgsWLIBWq4Ver4evry/+85//1HaVSMWWLFkCrVYLHxLSgiEAABswSURBVB8fREZGoqSkRJk3ceJENGrUqNp1Fy5cCDc3N3h4eGDPnj1PorqKb7/9Fh4eHnBzc8N7771XZX5GRgZCQ0Oh1+sRHByM7OxsZd6MGTOg1Wrh5eWFSZMmQURQXFyMPn36wNPTE1qtFrNmzapS5tatW6HRaJCQkAAASE9PR4MGDeDr6wtfX1/86U9/Upa9e/cuXn/9dbRr1w6enp746quvAAAHDx6Ev78/rK2tsXXr1irPUVBQAEdHR0yYMEGZdvLkSeh0Ori5uSn1BYA5c+Yo+3lYWBguX74MADhw4ADs7OyUev3973+/73bbt28f/P394evri6CgIFy6dAkAMGXKFKWcdu3a4ZlnngEA7N+/X5nu6+sLGxsb7NixAwCwfPlyuLm5QaPRIDc3t8bXkZ4u99vvMjMzERISAj8/P+j1euzevbvK/EaNGmHx4sVPqsqPNFasX78e7u7ucHd3x/r165XpwcHB8PDwUPr/tWvXlHn/+te/4O3tDa1Wi5deekmZnpmZibCwMHh5ecHb2xvp6ekAgLS0NHTu3Bnu7u4YNmwY7t69CwD45z//CZ1Op+yTycnJAIBNmzaZ7XtWVlZISkpCYWGh2fQWLVpg8uTJNZZV0xgWHR0NnU4HvV6P8PDwKvvy4sWLzfbxmzdvIiIiAnq9Hp06dcK5c+fu+xqICGbPno127drBy8sLH3/88YO8pL+OiFT7r2I2PUlHjhyRgIAAKSkpERGR69evS05OTi3Xyhz7hXpkZ2eLs7OzFBcXi4jIkCFDZO3atSIicuLECRkxYoQ0bNjQ4rrnz58XvV4vJSUl8tNPP4mrq6sYjcZfXZeH6RdGo1FcXV0lNTVVSktLRa/Xy/nz582WGTx4sKxbt05ERPbt2ycjRowQEZHDhw/L888/L0ajUYxGowQEBMj+/fulqKhIYmNjRUSktLRUgoKCZPfu3Up5BQUF0qVLF+ncubOcOHFCRETS0tJEq9VarOPbb78ts2fPFhGR8vJyuX79urLO6dOnZeTIkbJly5Yq602aNEkiIyNl/PjxyrSOHTvKkSNHxGQySXh4uFKv/Px8ZZlly5bJuHHjRERk//790qdPn4fabu7u7pKcnCwiIitWrJBRo0ZVWf/jjz+WMWPGVJmel5cnTZs2laKiIhEROXXqlKSlpUmbNm2Udv9aHC/U40H2u9dee00+/fRTEakYI9q0aWM2f+DAgTJ48GBZtGjRI9XlQfvFo4wVeXl54uLiInl5eXLjxg1xcXGRGzduiIhIt27dlHHgXikpKeLr66ssd/XqVWVet27dZO/evSIiUlhYqOwvQ4YMkejoaBERGTdunLL97t2/Y2JipGfPnlWe78yZM+Li4mKx7f7+/hIXF1djWdWNYWVlZdKyZUtl/50+fbrMnTtXmZ+ZmSlhYWHSunVrZZlp06bJvHnzRETkwoUL0r17dxGp+TVYs2aNjBw5UsrLy6tsr1/jv/3CYiblEVaVuXLlClq0aIH69esDAFq0aAEHBwc4Oztj5syZ6NSpEzp16qQcPfn666/RuXNn+Pn54cUXX8TVq1cBAPPmzcOoUaMQFhYGZ2dnbNu2DTNmzIBOp0N4eDjKyspqrY30eBmNRty5cwdGoxHFxcVwcHBAeXk5pk+fjg8++KDa9WJiYjB8+HDUr18fLi4ucHNzw/Hjx59InY8fPw43Nze4urqiXr16GD58OGJiYsyWSU5ORmhoKAAgJCREma/RaFBSUoK7d++itLQUZWVleO6552Bra4uQkBAAQL169eDv7292pGXOnDmYMWMGbGxsHqiOa9aswV//+lcAgJWVFVq0aAEAcHZ2hl6vh5VV1eHz5MmTuHr1KsLCwpRpV65cQUFBAQIDA6HRaPDKK68oRzKbNGmiLFdUVHTfX/6pabtpNBoUFBQAAPLz8+Hg4FBl/ejoaERGRlaZvnXrVvTq1Qu2trYAAD8/Pzg7O9dYF3r6PMh+V1M/2rFjB1xdXaHValVV5+rGij179qBHjx5o1qwZmjZtih49euDbb7+t8flWrVqF8ePHo2nTpgCAZ599VnkOo9GIHj16AAAaNWoEW1tbiAhiY2MxePBgAMCoUaMeav+ubp+8ePEirl27hi5dujxwWfeqDHlFRUUQERQUFJi9llOmTMEHH3xgVs6929HT0xPp6em4evVqja/BypUr8fbbbyvjYeX2+i0wsKpMWFgYsrKy0K5dO7zxxhuIi4tT5jVp0gTHjx/HhAkTlK8JgoKCcOzYMSQmJmL48OFmASU1NRW7du1CTEwMRowYgZCQEJw9exYNGjTArl27nnjb6PFzdHTEtGnT0Lp1a9jb28POzg5hYWFYvnw5+vfvD3t7+2rXzcnJQatWrZTHTk5OyMnJeRLVfqDnNhgMytfw27dvR2FhIfLy8hAYGIiQkBDY29vD3t4ePXv2hJeXl9m6t27dwtdff60MvomJicjKykLfvn2r1CUtLQ1+fn7o1q0b4uPjlfWBipDr7++PIUOGKB8Gq2MymfDmm29i0aJFVdrq5ORUbVtnz56NVq1aYdOmTWZf/R89ehQGgwG9evXC+fPn77vdPv/8c/Tu3RtOTk7YuHFjlVMiMjIykJaWhu7du1ep++bNmy2+adL/LQ+y382bNw9ffPEFnJyc0Lt3b3zyyScAKkLS+++/j7lz56quztWNFfdbd8yYMfD19cW7776rnKaTkpKClJQUvPDCCwgICFACbkpKCp555hkMHDgQfn5+mD59OsrLy5GXl4dnnnkG1tbWFp9jxYoVaNu2LWbMmGHx6/Ivv/zS4r4XHR2NYcOGmQXK6sqyNIbVrVsXK1euhE6ng4ODA5KTkxEVFQUA2LlzJxwdHWEwGKpsx23btgGo+KCQkZGB7OzsGrdjamoqvvzyS3To0AG9evXCxYsXq7TlcWFgVZlGjRrh5MmT+J//+R+0bNkSw4YNw7p16wBA6dSRkZE4evQoACA7Oxs9e/aETqfDokWLlDc2AOjVqxfq1q0LnU6H8vJyhIeHAwB0Op1y7g093W7evImYmBikpaXh8uXLKCoqwoYNG7BlyxZMnDixxnXFws8iPqnf9n6Q5168eDHi4uLg5+eHuLg4ODo6wtraGpcuXcKFCxeUgTQ2NhYHDx5U1jMajYiMjMSkSZPg6uoKk8mEKVOm4MMPP6zynPb29sjMzERiYiI++ugjvPTSSygoKIDRaER2djZeeOEFnDp1CoGBgZg2bVqNbfr000/Ru3dvs4H9Qdq6YMECZGVl4eWXX8by5csBAP7+/sjIyMDp06cxceJEDBgw4L5lLVmyBLt370Z2djbGjBmDqVOnmi23efNmDB48GHXq1DGbfuXKFZw9exY9e/assX309HuQ/S46OhqjR49GdnY2du/ejZEjR8JkMmHu3LmYMmVKjefE/xYeZayoad1Nmzbh7NmziI+PR3x8PDZu3AigYvy4ePEiDhw4gOjoaLz66qu4desWjEYj4uPjsXjxYpw4cQI//fQT1q1bd9/6jR8/HqmpqXj//fcxf/58s+X+85//wNbWFj4+PlXKsPQh0lJZ1Y1hZWVlWLlyJRITE3H58mXo9XosXLgQxcXFWLBggdmH40qzZs3CzZs34evri08++QR+fn733Y6lpaWwsbFBQkICXnvtNYwdO7bKso8LA6sK1alTB8HBwXjnnXewfPly5ZPjvTtB5d8TJ07EhAkTcPbsWXz22WdmF9xUnlZgZWWFunXrKutYWVnBaDQ+qebQb+j777+Hi4sLWrZsibp162LgwIGYO3cuLl26BDc3Nzg7O6O4uBhubm5V1nVyckJWVpbyODs72+LXyL+FB3luBwcHbNu2DYmJiViwYAEAwM7ODtu3b0dAQAAaNWqERo0aoVevXjh27Jiy3uuvvw53d3flW4jCwkKcO3cOwcHBcHZ2xrFjx9C/f38kJCSgfv36aN68OQCgffv2aNu2LVJSUtC8eXPY2toiIiICADBkyBCcOnWqxjYdPXoUy5cvh7OzM6ZNm4YNGzZg1qxZcHJyMjs1obrt/NJLLyn7epMmTZRg0Lt3b5SVlSE3N7fa7Xb9+nWcPn0anTt3BgAMGzYMR44cMSu/uqOo//rXvxAREYG6devW2D56+j3Ifrd69WoMHToUABAYGIiSkhLk5ubiP//5D2bMmAFnZ2csXboU//jHP5QPWLVd5+rGiprWdXR0BAA0btwYL730knI6lJOTE/74xz+ibt26cHFxgYeHBy5evAgnJyf4+fnB1dUV1tbWGDBgAE6dOoUWLVoogba6+gHA8OHDlVMFKlW3T54+fRpGoxHt27e3uE3uLau6MSwpKQkA0LZtW2g0GgwdOhRHjhxBamoq0tLSYDAY4OzsjOzsbPj7++Pnn39GkyZNsHbtWiQlJWHDhg24fv06XFxcatyOTk5OGDRoEAAgIiICZ86csVjnx4GBVWV+/PFHs0PqSUlJaNOmDYCKrw4q/w8MDARQcY5R5Y537xWQ9PvQunVrHDt2DMXFxRAR7Nu3D1OnTsXPP/+M9PR0pKenw9bWVjnn+V79+/fH5s2bUVpairS0NFy8eBGdOnV6IvXu2LEjLl68iLS0NNy9exebN29G//79zZbJzc2FyWQCUHE3g8pP7q1bt0ZcXByMRiPKysoQFxennBLw1ltvIT8/H0uXLlXKsbOzQ25urrI9AgICsHPnTnTo0AHXr19HeXk5AOCnn37CxYsX4erqCo1Gg379+uHAgQMAKq7A9/b2rrFNmzZtQmZmJtLT07F48WK88soreO+992Bvb4/GjRvj2LFjEBFs2LABf/zjHwHAbF/fuXMnPD09AQA///yzclTj+PHjMJlMaN68ebXbrWnTpsjPz0dKSgoA4LvvvjM7TeLHH3/EzZs3lXHjXtWdQ0f/9zzIfte6dWvs27cPAHDhwgWUlJSgZcuWiI+PV/ahyZMn429/+5vZnTBqs87VjRU9e/bE3r17cfPmTdy8eRN79+5Fz549YTQalSvjy8rK8M033yhHOQcMGID9+/cr5aakpMDV1RUdO3bEzZs3cf36dQBAbGwsvL29odFoEBISotwxZP369Rb37127dsHd3V15bDKZsGXLFgwfPrxKmy3tk9WVVd0Y5ujoiOTkZKW+lWOCTqfDtWvXlNfSyckJp06dwh/+8AfcunVLucPB559/jq5du6JJkyY1vgYDBgxAbGwsACAuLg7t2rWr+QV9FNVdjSW8S0CtSEhIkMDAQPHy8hKdTicRERFy/fp1adOmjcybN086deokHTp0kIsXL4qIyI4dO8TFxUWCgoJk2rRp0q1bNxERmTt3rtlVnPdeKf7LeQ+L/UJd3n77bfHw8BCtVisjRoxQ7jBR6d7XPiYmRubMmaM8nj9/vri6ukq7du3Mrqj/NR62X+zatUvc3d3F1dVV5s+fLyIic+bMkZiYGBER2bJli7i5uYm7u7tERUUp7TIajfL666+Lp6eneHl5yZQpU0REJCsrSwCIp6enGAwGMRgMsmrVqirPe+/VwVu3bhVvb2/R6/Xi5+cnO3fuVJZLT0+XLl26iE6nk+7du0tGRoaIiBw/flwcHR3F1tZWmjVrJt7e3lWeY+3atWZ3CThx4oRotVpxdXWV8ePHi8lkEpGKK661Wq3odDrp27evZGdni4jIJ598otSrc+fOcvjw4Rq3m4jItm3bxMfHR/R6vXTr1k1SU1OVeXPnzpWZM2dWqWdaWpo4ODgoV/hWWrZsmTg6OkqdOnXE3t5eoqKiqqz7oDheqMv99rvz58/L888/L3q9XgwGg+zZs6dKGY/6HiLycP3i144VIiKrV6+Wtm3bStu2bWXNmjUiInL79m3x9/cXnU4n3t7eMmnSJOUOKSaTSaZMmSJeXl7i4+OjXP0vIrJ3717R6XTi4+Mjo0aNktLSUhERSU1NlY4dO0rbtm1l8ODByvNPmjRJvL29xWAwSHBwsJw7d04pa//+/dK5c2eL7XVxcZELFy6YTauurJrGsJUrV4qnp6cyvuTm5lZ5rnvvBHLkyBFxc3MTDw8PiYiIUO6UUN1rICJy8+ZN6d27t/j4+EhAQIAkJSVZbNODQg13CdCIhXMTKmk0GqlpPj05zs7OSEhIUK5Urk0ajcbiOS30+8Z+QZawX5Al7BdkyX/7hcWLKXhKABERERGpGo+w0kPjJ2OyhP2CLGG/IEvYL8gSHmElIiIioqcWAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqZp1TTNtbGxMGo2GoZbM2NjYPLHfnKenB/sFWcJ+QZawX5AlNjY2purm8bZW9NB4OxKyhP2CLGG/IEvYL8gS3taKiIiIiJ5aDKxEREREpGoMrERERESkagysRERERKRqDKxEREREpGoMrERERESkagysKqLRaDBy5EjlsdFoRMuWLdG3b9+HKsfZ2Rm5ubmPvAypy7Jly+Dj4wOtVoulS5cCALZs2QKtVgsrKyskJCRYXC8rKwshISHw8vKCVqvFsmXLlHnz5s2Do6MjfH194evri927dz+RthARET2MGn84gJ6shg0b4ty5c7hz5w4aNGiA7777Do6OjrVdLVKBc+fOYdWqVTh+/Djq1auH8PBw9OnTBz4+Pti2bRvGjRtX7brW1tb48MMP4e/vj8LCQrRv3x49evSAt7c3AGDKlCmYNm3ak2oKERHRQ+MRVpXp1asXdu3aBQCIjo5GZGSkMu/GjRsYMGAA9Ho9AgICcObMGQBAXl4ewsLC4Ofnh3HjxpndjPmLL75Ap06d4Ovri3HjxqG8vPzJNogeiwsXLiAgIAC2trawtrZGt27dsH37dnh5ecHDw6PGde3t7eHv7w8AaNy4Mby8vJCTk/Mkqk1ERPRYMLCqzPDhw7F582aUlJTgzJkz6Ny5szJv7ty58PPzw5kzZ/CPf/wDr7zyCgDgnXfeQVBQEBITE9G/f39kZmYCqAg5X375JQ4fPoykpCTUqVMHmzZtqpV20aPx8fHBwYMHkZeXh+LiYuzevRtZWVkPXU56ejoSExPN+tXy5cuh1+sxduxY3Lx583FWm4iI6LFgYFUZvV6P9PR0REdHo3fv3mbzDh06pJzj2r17d+Tl5SE/Px8HDx7EiBEjAAB9+vRB06ZNAQD79u3DyZMn0bFjR/j6+mLfvn346aefnmyD6LHw8vLCzJkz0aNHD4SHh8NgMMDa+uHO6Ll9+zYGDRqEpUuXokmTJgCAP//5z0hNTUVSUhLs7e3x5ptv/hbVJyIieiQ8h1WF+vfvj2nTpuHAgQPIy8tTplv63WWNRmP2/71EBKNGjcLChQt/u8rSExMVFYWoqCgAwN/+9jc4OTk98LplZWUYNGgQXn75ZQwcOFCZ/txzzyl/v/baaw99gR8REdGTwCOsKjR27Fi8/fbb0Ol0ZtO7du2qfKV/4MABtGjRAk2aNDGb/u9//1v5Wjc0NBRbt27FtWvXAFScA5uRkfEEW0KPU+XrmJmZiW3btpmd31wTEUFUVBS8vLwwdepUs3lXrlxR/t6+fTt8fHweX4WJiIgeEwZWFXJycsJf/vKXKtPnzZuHhIQE6PV6zJo1C+vXrwdQcW7rwYMH4e/vj71796J169YAAG9vb8yfPx9hYWHQ6/Xo0aOHWUChp8ugQYPg7e2Nfv36YcWKFWjatCm2b98OJycnHD16FH369EHPnj0BAJcvX1ZOKTl8+DA2btyI2NjYKrevmjFjBnQ6HfR6Pfbv348lS5bUWvuIiIiqo7H0NbMyU6ORmubT75NGo7F4egL9vrFfkCXsF2QJ+wVZ8t9+UfUcR/AIKxERERGpHAMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpWo0/zWpjY3NVo9E8V9My9PtjY2Nj0mg0/LBDZtgvyBL2C7KE/YIssbGxuVrdvBp/OICIiIiIqLbx0w0RERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGpGgMrEREREakaAysRERERqRoDKxERERGp2v8H7o2hdpPdVfwAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data = [[ \"Ham\", accuracy_ham,  precision_ham, recall_ham,  F1measure_ham],\n",
    "        [ \"Spam\", accuracy_spam,  precision_spam,  recall_spam, F1measure_spam],\n",
    "        [ \"Model\", accuracy_model,  \"\",\"\"  ,\"\" ]]\n",
    "\n",
    "fig, axs =plt.subplots(2,1)\n",
    "collabel=(\"Class\",\"Accuracy\", \"Precision\", \"Recall\",\"F1-measure\")\n",
    "axs[0].axis('tight')\n",
    "axs[0].axis('off')\n",
    "axs[1].axis(\"off\")\n",
    "the_table = axs[0].table(cellText=data,colLabels=collabel,loc='center')\n",
    "the_table.auto_set_font_size(False)\n",
    "the_table.set_fontsize(10)\n",
    "the_table.scale(2, 2)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

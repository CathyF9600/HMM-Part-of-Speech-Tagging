import os
import sys
import argparse
import re
import numpy as np
import time
global ALPHA 
ALPHA = 1e-7

def initial_probabilities(tag_list):
    initial_probs = {}
    total_tags = len(tag_list)
    for tag in tag_list:
        if tag[0] in initial_probs:
            initial_probs[tag[0]] += 1
        else:
            initial_probs[tag[0]] = 1
    for tag in initial_probs:
        initial_probs[tag] = initial_probs[tag] / total_tags
    value_list = [(key, initial_probs[key]) for key in sorted(initial_probs.keys())]
    matrix = np.array(value_list)

    return initial_probs


def transition_probabilities(TAGS, tag_lists):
    ordered_transition = np.zeros((len(TAGS), len(TAGS)))
    tag_counts = np.zeros(len(TAGS))
    for i in range(len(tag_lists)-1):
        current_tag = tag_lists[i]
        next_tag = tag_lists[i+1]
        prev_idx = np.where(TAGS == current_tag)
        next_idx = np.where(TAGS == next_tag)
        if next_tag == "NP0":
            ordered_transition[prev_idx, next_idx] += 55
        if next_tag == "AJ0":
            ordered_transition[prev_idx, next_idx] += 35
        if next_tag == "AV0":
            ordered_transition[prev_idx, next_idx] += 25
        if next_tag == "NN1":
            ordered_transition[prev_idx, next_idx] += 25
        if next_tag == "NN2":
            ordered_transition[prev_idx, next_idx] += 25
        if next_tag == "VVD":
            ordered_transition[prev_idx, next_idx] += 10
        if next_tag == "VVG":
            ordered_transition[prev_idx, next_idx] += 10
        if next_tag == "VVN":
            ordered_transition[prev_idx, next_idx] += 10
        else:
            ordered_transition[prev_idx, next_idx] += 1
        tag_counts[np.where(TAGS == tag_lists[i])] += 1
    for i in range(len(tag_counts)):
        if tag_counts[i] == 0:
            tag_counts[i] = 1
    for i in range(len(TAGS)):
        ordered_transition[i, :] = ordered_transition[i,:] / tag_counts[i]

    return ordered_transition


def observation_probabilities(word_list, tag_list, TAGS, total_word):
    observation_prob = np.zeros((len(TAGS), len(total_word)))
    
    for i in range(len(tag_list)):
        tag_index = np.where(np.array(TAGS) == tag_list[i])
        word_index = np.where(total_word == word_list[i])
        observation_prob[tag_index, word_index] += 1
    
    tag_counts = observation_prob.sum(axis=1)
    tag_counts[tag_counts == 0] = 1
    observation_prob = observation_prob / tag_counts.reshape(len(TAGS),1)
    
    return observation_prob


def viterbi1(E, S, I, T, M, unique_set): # broadcast, fast
    """
    E: test sentences
    S: TAGS
    """
    prob = np.zeros((len(S), len(E))) + ALPHA
    prev = np.full((len(S), len(E)), None)
    # Determine values for time step 0
    T = T.T
    j = 0
    if (E[j] in unique_set):
        word_id = np.where(unique_set == E[j])
        prob[:,j] = I * M[:, word_id].squeeze()
    else:
        prob[:,j] = I
    for i in range(1,len(E)):
        prev_prob = prob[:, i-1]
        # print(T.shape)
        result = T * prev_prob.reshape(1, -1)
        if (E[i] in unique_set):
            word_id = np.where(unique_set == E[i])
            multiplier = M[:,word_id].squeeze() # (91,)
            result = result * multiplier.reshape(len(S),1) # (91,91) * (91,1)
        else:
            result = result
        max_elements = np.amax(result, axis=1)
        max_indices = np.argmax(result, axis=1)
        prob[:,i] = max_elements
        prev[:,i] = max_indices

    return prob.T, prev.T


def viterbi2(E, S, I, T, M): # non broadcasting, slow
    global rule_to
    prob = np.zeros((len(E), len(S)))
    prev = np.zeros((len(E), len(S)))
    # Determine values for time step 0
    # print(E.index(E[0]))
    for i in range(len(S)):
        prob[0][i] = I[i] * M.get(S[i], {}).get(E[0], 1e-10) # 1 if unseen else M
        prev[0][i] = None

    for t in range(1,len(E)):
        if E[i] in ["to", "To"]:
            prob[t] 
        else:
            for i in range(len(S)):
                max_prob = 0
                max_idx = 0
                for j in range(len(S)):
                    p = prob[t-1][j] * T[j, i] * M.get(S[i], {}).get(E[t], 1e-10)
                    # print(p)
                    if p > max_prob:
                        max_prob = p
                        max_idx = j
                prob[t][i] = max_prob
                prev[t][i] = max_idx

    return prob, prev


def read_train_file(train_files):
    words = []
    labels = []
    for filename in train_files:
        f = open(filename)
        lines = f.readlines()
        for s in lines:
            if(s[0] == ':'):
                a = ':'
                b = s.split(':')[-1]
            else:
                a, b = s.split(':')
            words.append(a.strip())
            labels.append(b.strip())
        f.close()

    return words, labels


def split_into_sentences_train(words, tags):
    sentence_words = []
    sentence_tags = []
    current_sentence_words = []
    current_sentence_tags = []

    for i in range(len(words)):
        word = words[i]
        tag = tags[i]

        # If word is not end of sentence punctuation, add it to the current sentence
        if not word.endswith((".", "?", "!")):
            current_sentence_words.append(word)
            current_sentence_tags.append(tag)

        # If word is end of sentence punctuation, add the current sentence to the list of sentences
        # and start a new current sentence
        else:
            current_sentence_words.append(word)
            current_sentence_tags.append(tag)
            sentence_words.append(current_sentence_words)
            sentence_tags.append(current_sentence_tags)
            current_sentence_words = []
            current_sentence_tags = []

    # If there is a current sentence at the end, add it to the list of sentences
    if current_sentence_words:
        sentence_words.append(current_sentence_words)
        sentence_tags.append(current_sentence_tags)

    return sentence_words, sentence_tags


def read_test_file(test_file):
    words = []
    f = open(test_file, "r")
    lines = f.readlines()
    for s in lines:
        words.append(s.strip())     
    f.close()
    
    return words


def split_into_sentences_test(words):
    sentences = []
    current_sentence = []

    for word in words:
        current_sentence.append(word)
        if word.endswith(".") or word.endswith("?") or word.endswith("!"):
            sentences.append(current_sentence)
            current_sentence = []

    # add last sentence if not empty
    if current_sentence:
        sentences.append(current_sentence)

    return sentences


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    training_list = args.trainingfiles[0]

    print("Starting the tagging process.")
    
    start = time.time()
    TAGS = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
            "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
            "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
            "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
            'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
            'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
            'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
            'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
            'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']
    
    punc_pos = {
        '.': 'PUN', 
        ',': 'PUN', 
        '?': 'PUN', 
        '-': 'PUN', 
        ':': 'PUN', 
        '!': 'PUN', 
        ';': 'PUN', 
        '"': 'PUQ', 
        '(': 'PUL', 
        '[': 'PUL',
        '{': 'PUL',
        ')': 'PUR',
        ']': 'PUR',
        '}': 'PUR'
        }

    # python3 tagger.py --trainingfiles /Users/yuchunfeng/Documents/CSC384/A4/training1.txt --testfile /Users/yuchunfeng/Documents/CSC384/A4/CH3.txt --outputfile /Users/yuchunfeng/Documents/CSC384/A4/output.txt

    train_words, train_labels = read_train_file(training_list)
    x_train, y_train = split_into_sentences_train(train_words, train_labels)

    test_words = read_test_file(args.testfile)
    x_test = split_into_sentences_test(test_words)

    unique_words = np.unique(train_words)
    
    initial_prob = initial_probabilities(y_train)
    transition_prob = transition_probabilities(np.array(TAGS), train_labels) + ALPHA
    observation_prob = observation_probabilities(train_words, train_labels, TAGS, unique_words) + ALPHA

    correct = 0
    total = 0
    calc = time.time()

    # convert initial matrix to numpy array
    initial_prob_matrix = np.zeros(len(TAGS)) + ALPHA
    for i, tag1 in enumerate(TAGS):
        initial_prob_matrix[i] = initial_prob.get(tag1, 0)

    output_filename = args.outputfile
    f = open(output_filename, "a")
    f.seek(0)
    f.truncate()

    pnp = ["i", "me", "my", "mine", "you", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs"]
    dps = ["my", "his", "her", "its", "our", "their"]
    for i in range(len(x_test)): 
        sentence = x_test[i]
        idx_to_tag = {i: tag for i, tag in enumerate(TAGS)}
        pred_tags = []

        prob, prev = viterbi1(sentence, TAGS, initial_prob_matrix, transition_prob, observation_prob, unique_words)        
        end_tag_idx = np.argmax(prob[-1])

        # Traverse prev list from end to beginning to get predicted tags
        for a in range(len(sentence) - 1, -1, -1):
            pred_tags.append(idx_to_tag[end_tag_idx])
            if(a!=0):
                end_tag_idx = int(prev[a][end_tag_idx])
        pred_tags.reverse()

        # Hardcoded prediction
        for w in range(len(sentence)):
            s = sentence[w]
            if s in punc_pos:
               pred_tags[w] = punc_pos[s] 
            elif s in ["of", "Of"]:
                pred_tags[w]= "PRF"
            elif s in ["am", "are", "'m", "'re"]:
                pred_tags[w]= "VBB"
            elif s in ["was", "were"]:
                pred_tags[w]= "VBD"
            elif s in ["was", "were"]:
                pred_tags[w]= "VBD"
            elif s == "been":
                pred_tags[w]= "VBN"
            elif s == "is":
                pred_tags[w]= "VBZ"
            elif s == "did":
                pred_tags[w]= "VDD"
            elif s == "doing":
                pred_tags[w]= "VDG"
            elif s == "done":
                pred_tags[w]= "VDN"
            elif s == "does":
                pred_tags[w]= "VDZ"
            elif s in ["'ve"]:
                pred_tags[w]= "VHB"
            elif s == "has":
                pred_tags[w] = "VHZ"
            elif s in ["would", "could", "'ll"]:
                pred_tags[w] = "VM0"
            elif s in ["not", "n't", "nae"]:
                pred_tags[w] = "XX0"
            elif s == "have":
                if pred_tags[w-1] == "TO0":
                    pred_tags[w] = "VHI"
            elif s.lower() in ["why", "how","wherever"]:
                pred_tags[w] = "AVQ"
            elif s[-2:] == "ly" and not s[0].isupper() and pred_tags[w] not in ["AJ0", "VVI"]:
                pred_tags[w] = "AV0"
            elif s.lower() in pnp and s.lower() not in dps and pred_tags[w] != "PNP":
                print("hi",s)
                pred_tags[w] = "PNP"
            elif s.lower() not in pnp and s.lower() in dps and pred_tags[w] != "DPS":
                print(s)
                pred_tags[w] = "DPS"
            
        for k in range(len(pred_tags)):
            f.write(x_test[i][k])
            f.write(" : ")
            f.write(pred_tags[k])
           
            f.write("\n")
    
    end = time.time()
    timing = end-start
    print("Time:",timing)
    print("\n")


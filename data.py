import sys
import os
import numpy as np


def load_data(filepath, num_speakers):
    f = open(filepath, "r")
    first = True
    block_count = 0
    digit_count = 0
    speaker_count = 0
    utterance_data = []
    block_data = []
    speaker_data = []
    digit_data = []
    for x in f:
        data = [float(item) for item in x.split()]
        
        # end of utterance
        if len(data) == 0:
            # add to block
            if first:
                first = False
            else:
                block_data.append(np.array(utterance_data))
                block_count += 1
                
                # add to speaker
                if block_count % 10 == 0:
                    speaker_data.append(block_data)
                    speaker_count += 1
                    
                    # add to digit
                    if speaker_count % num_speakers == 0:
                        digit_data.append(speaker_data)
                        digit_count += 1

                        # reset speaker data after adding to digit
                        speaker_data = []

                    # reset block data after adding to speaker
                    block_data = []

                # reset utterance data after adding to block
                utterance_data = []
        else:
            utterance_data.append(data)

    block_data.append(utterance_data)
    block_count += 1
    if block_count % 10 == 0:
        speaker_data.append(block_data)
        speaker_count += 1
                    
        # add to digit
        if speaker_count % num_speakers == 0:
            digit_data.append(speaker_data)
            digit_count += 1
    f.close()

    digit_data = np.array(digit_data, dtype=object).reshape((-1,))

    labels = []

    for digit in range(10):
        for i in range(num_speakers * 10):
            labels.append(digit)

    return digit_data, np.array(labels)


def load_train_data():
    return load_data("spoken_arabic_digit/Train_Arabic_Digit.txt", 66)


def load_test_data():
    return load_data("spoken_arabic_digit/Test_Arabic_Digit.txt", 22)


if __name__ == "__main__":
    test_digits, test_labels = load_train_data()
    print(test_digits.shape)
    print(test_digits[0].shape)
    print(test_labels.shape)
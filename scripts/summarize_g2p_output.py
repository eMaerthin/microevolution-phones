import fire
import numpy as np
import pandas as pd

def try_summarize(input_file, verbosity=1, *filter_letters):
    if filter_letters is ():
        filter_letters = ['A', 'E', 'I', 'O', 'U', 'Y'] # if you want to not filter at all, pass ''
    else:
        try:
            filter_letters = [x.upper() for x in filter_letters]
        except AttributeError:
            print(f'AttributeError - check if {filter_letters} is a tuple containing strings ONLY')

    try:
        summarize(input_file, verbosity, filter_letters)
    except IOError:
        print(f'Error while summarizing {input_file}')

def check_elem(elem, criteria):
    return any([elem.find(crit) != -1 for crit in criteria])

def summarize(input_file, verbosity, filter_letters):
    phonemes_temp=[]
    with open(input_file) as f:
        for line in f:
            for word in line.split():
                phonemes_temp.append(word)
    sorted_phonemes = np.sort(phonemes_temp)
    phonemes_filtered = list(filter(lambda x: check_elem(x, filter_letters), sorted_phonemes))
    unique_phonemes, unique_phonemes_counts = np.unique(phonemes_filtered, return_counts = True)

    if verbosity > 0:
        print(f'Total number of phonemes: {len(phonemes_filtered)} - number of unique phonemes: {len(unique_phonemes)}')
        print(pd.get_dummies(phonemes_filtered).sum()/len(phonemes_filtered))


if __name__ == '__main__':
    fire.Fire({
              'summarize-g2p': try_summarize
              })

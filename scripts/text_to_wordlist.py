import fire

def text_to_wordlist(input_file, output_file, verbosity=0):
    with open(output_file, 'w') as out:
        with open(input_file) as f:
            for line in f:
                for word in line.split():
                    lower = word.lower()
                    if verbosity>0:
                        print(lower)
                    out.write(lower+'\n')


if __name__ == '__main__':
    fire.Fire({
              'text2wordlist': text_to_wordlist,
              'text_to_wordlist': text_to_wordlist,
              })

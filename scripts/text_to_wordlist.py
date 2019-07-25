import logging
import fire
logger = logging.getLogger()

def text_to_wordlist(input_file, output_file, verbosity=0):
    with open(output_file, 'w') as out:
        with open(input_file) as f:
            for line in f:
                for word in line.split():
                    lower = word.lower()
                    logger.debug(lower)
                    out.write(lower+'\n')


if __name__ == '__main__':
    from logging.config import fileConfig
    fileConfig('logging.conf')
    fire.Fire({
              'text2wordlist': text_to_wordlist,
              'text_to_wordlist': text_to_wordlist,
              })

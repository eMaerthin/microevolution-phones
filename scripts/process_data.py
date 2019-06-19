import ssl

import fire

from audio_processors import process_playlist_url
from chain_runner import ChainRunner

ssl._create_default_https_context = ssl._create_stdlib_context

if __name__ == '__main__':
    """
    This is entry point of the library
    
    - 'playlist' option should be used to populate (file-based) database 
    based on youtube playlist url
    
    example use: 
    1) playlist
    2) run --dataset_home_dir '/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/dummy_test/' --verbose 1 process-chains
    3) run --dataset_home_dir '/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/dummy_test/' --verbose 1 process-chain --chain_name Formants
    4) run from_experiment_config --experiment-config-path '/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/dummy_test/_configs/test-config.json' process-chains
    """
    fire.Fire({
              'playlist': process_playlist_url,
              'run': ChainRunner
              })

# https://www.youtube.com/watch?v=-T7g8t1ojyk&list=PLQsLDm9Rq9bFZdgAABjtnjI1n5boQAuBd

# run from_experiment_config --experiment-config-path '/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/dummy_test/_configs/test-config.json' process-chains
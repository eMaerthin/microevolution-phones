import ssl

import fire

from audio_processors import process_playlist_url
from chain_runner import ChainRunner

ssl._create_default_https_context = ssl._create_stdlib_context

if __name__ == '__main__':
    """
    example use: 
    1) playlist --subjects_homedir '/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/chessnetwork/longPlaylist'
    2) run --dataset_home_dir '/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/dummy_test/' --verbose 1 process-chains
    """
    fire.Fire({
              'playlist': process_playlist_url,
              'run': ChainRunner
              })

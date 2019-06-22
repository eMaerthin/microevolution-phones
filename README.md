# microevolution-phones
This repo stores ideas and approaches to the microevolution of individual speaker's phonemes retrieved from recordings found on youtube

## Set-up
The project is written in python. It requires setting the environment using Pipenv. First ensure that `pipenv` is already installed - if not, type `pip install pipenv`.

Now, type `pipenv install` and `pipenv shell` et voila - you should be able to run scripts seamlessly now.

## Third party

- [Multicore-TSNE](#multicore-tsne)
- [pocketsphinx](#pocketsphinx)
- [pytube](#pytube)

### Multicore-TSNE
This is a multicore modification of Barnes-Hut t-SNE 
by L. Van der Maaten with python and Torch CFFI-based wrappers. 
This code also works faster than `sklearn.TSNE` on single core.

Officially distributed library has a defect 
that it occupies just a single core when run on macOS.
Hence, sources from macOS complaint fork are used: 
https://github.com/sg-s/Multicore-TSNE

(see [Issue #53: Using single core even when n_jobs=4 is used](https://github.com/DmitryUlyanov/Multicore-TSNE/issues/53))


### pocketsphinx
`PocketSphinx` is a lightweight speech recognition engine, 
specifically tuned for handheld and mobile devices, 
though it works equally well on the desktop.

While an official distribution of `pocketsphinx` is installed 
with `pipenv`, source repository contains an officially 
distributed generic US english acoustic 
model trained with latest `sphinxtrain`.

### pytube
A lightweight, dependency-free Python library 
(and command-line utility) for downloading YouTube 
Videos. https://python-pytube.readthedocs.io

Sometimes (like as of 2019-04-27) official distribution of `pytube` is not
working because youtube API changed in the meantime.
This will be patched officially for sure at some day 
but until then a forked repository with local fix is used.
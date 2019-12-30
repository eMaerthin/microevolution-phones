# Overview of praat scripts
1. [ crosscorelate ](#crosscorelate)
<a name="crosscorelate"></a>
## crosscorelate.praat
### Description
The aim of this script is to find offset between first and second input sound.
### Script signature
`./crosscorelate.praat input_sound_1 input_sound_2 start_time end_time`

The script requires two mandatory parameters: `input_sound_1` and `input_sound_2` (paths).
Furthermore two additional optional parameters can be provided: 
`start_time` (by default `0`) and `end_time` (by default `130`) - times are given in seconds.
`start_time` and `end_time` are used to improve running time performance of the optimal offset search using the crosscorelate algorithm for a price of the risk to not find the proper offset if it is outside of the `(start_time, end_time)` interval.

### Example use
`Praat --run ./crosscorelate.praat sound1.wav sound2.wav 0 100`

### Troubleshooting
If Praat is not found, type first `export PATH=$PATH_TO_PRAAT_DIR:$PATH`
for instance: `export PATH=/Applications/Praat.app/Contents/MacOS:$PATH`. 

See steps in https://gist.github.com/nex3/c395b2f8fd4b02068be37c961301caa7 to check how to set the environment variable permanently depending on the OS.

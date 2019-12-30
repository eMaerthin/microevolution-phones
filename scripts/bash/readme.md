# Overview of bash scripts
1. [ Praat_crosscorelate ](#praat)
<a name="praat"></a>
## praat_crosscorelate.sh
### Description
The aim of this script is to run `scripts/praat/crosscorelate.praat` on two given sets of files.
Wildcards can be used to filter files.

### Script signature
The script requires either 2 or 4 input arguments:
`./praat_crosscorelate.sh set1 set2`

or

`./praat_crosscorelate.sh set1 set2 start_time end_time`

If only two arguments are provided the script will use default values for `start_time` (`0`) and for `end_time` (`100`).

### Example use
`./praat_crosscorelate.sh 20190628/video/*819*.wav 20190628/TASCAM/test-*.mp3.wav` 

### Troubleshooting
If Praat is not found, type first `export PATH=$PATH_TO_PRAAT_DIR:$PATH`
for instance: `export PATH=/Applications/Praat.app/Contents/MacOS:$PATH`. 

See steps in https://gist.github.com/nex3/c395b2f8fd4b02068be37c961301caa7 to check how to set the environment variable permanently depending on the OS.

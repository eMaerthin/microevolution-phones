#!/bin/sh
# if Praat is not found, type export PATH=$PATH_TO_PRAAT_DIR:$PATH
# for instance: export PATH=/Applications/Praat.app/Contents/MacOS:$PATH

if (( $# != 2 )) && (( $# != 4 )); then
    echo "Illegal number of arguments. Expected 2 or 4. Passed" $#
    exit
fi

set1=$1
#/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/goats/raw_videos/20190628/video/*819*.wav
set2=$2
#/Volumes/Transcend/phd/microevolution-lang-phones-data/subjects/goats/raw_videos/20190628/TASCAM/test-DR-100_0029.mp3.wav

if [[ $set1 != /* ]]; then
    set1=$PWD/$set1
fi

if [[ $set2 != /* ]]; then
    set2=$PWD/$set2
fi


start_time=0
end_time=100

if (( $# == 4 )); then
    start_time=$3
    end_time=$4
fi

for j in $set1;
    do
    for k in $set2;
    do
        echo $j $k;
        Praat --run "$( dirname "${BASH_SOURCE[0]}" )"/../praat/crosscorelate.praat $j $k $start_time $end_time;
    done;
done

#!/bin/bash
#USER=$(whoami)
VAR=$(uname)
WORKDIR=$(pwd)

if [ $VAR == 'Darwin' ]
then
    brew install timidity abcmidi > /dev/null 2>&1
else
    apt-get install abcmidi timidity > /dev/null 2>&1
fi

(crontab -l; \
 echo "0 0 1 * * python3 $WORKDIR/train.py $WORKDIR/config.yaml" && \
 echo "0 10 * * * python3 $WORKDIR/predict.py $WORKDIR/config.yaml" \
) | crontab

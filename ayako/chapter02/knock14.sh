#!/bin/bash

#-nで行数指定，行数はコマンドライン引数から取得->$1
#行数=5とかで出力
head -n $1 popular-names.txt > output14.txt

exit 0
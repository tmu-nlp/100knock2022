#!/bin/bash

#-lで行数指定，行数はコマンドライン引数から取得->$1
#行数=800とかで出力
#-dで接頭辞の後ろに数字
#split -l 行数　-d ファイル名　分割後ファイルの接頭辞
split -l $1 -d popular-names.txt  output16-

exit 0
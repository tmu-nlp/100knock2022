#!/bin/bash

#sortで並び替え
#-rで逆順
#-nで数値を考慮した並べ替え
#-kで指定したフィールドで並べ替え,逆順の時は指定した数字の後ろにr,数値を考慮するときは後ろにn
sort -n -k 3rn popular-names.txt  > output18.txt

exit 0
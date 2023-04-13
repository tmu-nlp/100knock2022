#knock17
#1列目の文字列の種類（異なる文字列の集合）を求めよ．
#確認にはcut, sort, uniqコマンドを用いよ．

#!/bin/bash

#cutで一列目取り出し
#sortで文字列名簿順に並び替え
#uniqで重複している行を削除
cut -f 1 < popular-names.txt | sort | uniq > output17.txt

exit 0
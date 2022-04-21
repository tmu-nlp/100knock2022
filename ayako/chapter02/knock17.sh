#!/bin/bash

#cutで一列目取り出し
#sortで文字列名簿順に並び替え
#uniqで重複している行を削除
cut -f 1 < popular-names.txt | sort | uniq > output17.txt

exit 0
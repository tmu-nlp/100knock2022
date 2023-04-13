#!/bin/bash

#cutで一列目取り出し
#sortで文字列名簿順に並び替え
#uniqで重複している行を削除, -cで重複行をカウント
cut -f 1 < popular-names.txt | sort | uniq -c | sort -r > output19.txt

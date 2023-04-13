sort -k 3nr popular-names.txt > sorted-table-sh.txt

#-k オプション 場所と並べ替え種別を指定する
#「-k 2」なら2列目、「-k 2n」なら2列目を数値として並べ替える。
#結局-k 3nrは3列目を数値として降順に並べ替えている
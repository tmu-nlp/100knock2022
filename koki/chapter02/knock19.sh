cut -f 1 popular-names.txt | sort | uniq -c | sort -nr > res19.txt

#uniqコマンドにより重複した行を削除、cオプションにより重複した行数も表示する

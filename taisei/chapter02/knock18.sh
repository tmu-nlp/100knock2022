sort -r -t $'\t' -k 3 -n popular-names.txt > knock18_check.txt
#-k 3 だと数値を文字列としてsortしちゃう。-nをつけると数値としてsortしてくれる
diff knock18_output.txt knock18_check.txt
#↑同じ数値を持つものの表示がところどころ異なる
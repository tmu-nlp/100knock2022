cut -f 1 popular-names.txt > knock12_check_1.txt
diff col1.txt knock12_check_1.txt

cut -f 2 popular-names.txt > knock12_check_2.txt
diff col2.txt knock12_check_2.txt
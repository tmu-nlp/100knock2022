#!/bin/zsh

cut -f 1 popular-names.txt > col1.txt
sort -u col1.txt -o col1_sorted.txt
uniq col1_sorted.txt col1_uni.txt
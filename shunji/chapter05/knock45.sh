#!/bin/zsh

# uniqは連続して重複した行を削除するので，その前にsortを挟む
# uniq -c 重複した行数表示
# sort -n 数値としてソート
# sort -r 降順ソート
cat 45.txt | sort | uniq -c | sort -nr > 45sh1.txt
cat 45.txt | grep "行う" | sort | uniq -c | sort -nr > 45sh2.txt
cat 45.txt | grep "なる" | sort | uniq -c | sort -nr > 45sh3.txt
cat 45.txt | grep "与える" | sort | uniq -c | sort -nr > 45sh4.txt

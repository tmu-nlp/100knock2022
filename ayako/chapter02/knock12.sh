#!/bin/bash

#-fで何番目のフィールドを抜き出すか指定，区切りのデフォルトはタブ，-dで区切り文字指定可能
cut -f 1 < popular-names.txt > col1.txt | cut -f 2 < popular-names.txt > col2.txt

exit 0
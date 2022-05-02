#!/bin/zsh

# cut [option] [file]
# -f [切り出すフィールド]
# -d [区切り文字(デフォはタブ)]
cut -f 1 popular-names.txt > col1.txt
cut -f 2 popular-names.txt > col2.txt

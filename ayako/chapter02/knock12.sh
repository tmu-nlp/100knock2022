#knock12
#各行の1列目だけを抜き出したものをcol1.txtに，
#2列目だけを抜き出したものをcol2.txtとしてファイルに保存せよ．
#確認にはcutコマンドを用いよ．

#!/bin/bash

#-fで何番目のフィールドを抜き出すか指定，区切りのデフォルトはタブ，-dで区切り文字指定可能
cut -f 1 < popular-names.txt > col1.txt | cut -f 2 < popular-names.txt > col2.txt

exit 0
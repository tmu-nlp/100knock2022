sed 's/\t/ /g' popular-names.txt > replaced-sed.txt
#スクリプトコマンド : 's/置換前/置換後/g' gはglobalを表し、全行に渡って置換する
#/はデリミタ(区切り文字)

cat popular-names.txt | tr '\t' ' ' > ./replaced-tr.txt
# | はパイプ 左コマンドの標準出力を右コマンドが標準入力として受け取る
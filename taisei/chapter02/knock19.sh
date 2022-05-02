cut -f 1 popular-names.txt | sort | uniq -c | sort -r -t $'\t' -k 1 -n > knock19_check.txt

diff knock19_output.txt knock19_check.txt
#タブのせいで全行一致していない判定0420
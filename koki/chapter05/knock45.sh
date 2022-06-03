#頻出上位5位
cat ./result/output45.txt | sort | uniq -c | sort -nr | head -n 5 

#「行う」という動詞の格パターン頻出上位5位
cat ./result/output45.txt | grep "行う" | sort | uniq -c | sort -nr | head -n 5 

#「なる」という動詞の格パターン頻出上位5位
cat ./result/output45.txt | grep "なる" | sort | uniq -c | sort -nr | head -n 5 

#「与える」という動詞の格パターン頻出上位5位
cat ./result/output45.txt | grep "与える" | sort | uniq -c | sort -nr | head -n 5 

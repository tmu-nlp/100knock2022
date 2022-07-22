for N in `seq 1 20`
do
    fairseq-interactive --path result/checkpoint_best.pt --beam $N result/preprocessing < ./tokenized/test.ja | grep '^H' | cut -f3 > 94_$N.out
done
for N in `seq 1 20`
do
    fairseq-score --sys 94_$N.out --ref test.en > 94_$N.score
done
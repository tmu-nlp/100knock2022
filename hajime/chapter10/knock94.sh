for N in `seq 1 20` ; do
    fairseq-interactive --path ./checkpoint_best.pt --beam $N ./ < test.ja | grep '^H' | cut -f3 > 94.$N.out
done

for N in `seq 1 20` ; do
    fairseq-score --sys 94.$N.out --ref test.en > 94.$N.score
done

python3 knock94.py
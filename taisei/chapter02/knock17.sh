cut -f 1 popular-names.txt | sort | uniq > knock17_check.txt

diff knock17_output.txt knock17_check.txt
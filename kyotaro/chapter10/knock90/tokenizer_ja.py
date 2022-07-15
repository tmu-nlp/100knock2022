import MeCab
import unidic
from MeCab import Tagger
import re
import sys


def tokenize_ja(input_file, output_file):
    wakati = Tagger('-Owakati')
    num = re.compile(r'[0-9]+')
    # p = r'[\（\）\「\」\『\』\-\(\)\～]'
    with open(input_file, "r") as input_:
        with open(output_file, "w") as out_:
            for line in input_:
                line = line.strip()
                line = num.sub('0', line)
                # line = re.sub(p, '', line)
                result = wakati.parse(line).strip()
                out_.write(result)
                out_.write("\n")

input_file = sys.argv[1]
output_file = sys.argv[2]

tokenize_ja(input_file, output_file)
import re
import sys


def tokenize_en(input_file, output_file):
    num = re.compile(r'[0-9]+')
    with open(input_file, "r") as input_:
        with open(output_file, "w") as out_:
            for line in input_:
                line = line.strip()
                line = num.sub('0', line)
                out_.write(line)
                out_.write("\n")

input_file = sys.argv[1]
output_file = sys.argv[2]

tokenize_en(input_file, output_file)
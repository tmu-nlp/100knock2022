# 手間: 25
import gzip
import json
file_path = "../data/jawiki-country.json.gz"
# import sys
# file_path = sys.argv[1]


def read_uk_text():
    with gzip.open(file_path, "rt") as f:
        for line in f:
            x = json.loads(line)
            if x["title"] == "イギリス":
                return x["text"]


def main():
    uk_text = read_uk_text()
    print(uk_text)


if __name__ == "__main__":
    main()

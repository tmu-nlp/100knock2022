from knock20 import read_uk_text
import re


def main():
    uk_text = read_uk_text().split("\n")
    ans = []
    f = False
    ans = dict()
    d_key = ""
    for line in uk_text:
        if re.search(r"^\{\{基礎情報", line):
            f = True
            continue

        # q25.py に以下の5行を加えた
        line = re.sub(r"'{2,}", "", line)
        line = re.sub(
            r"\[\[(?!ファイル|Category)(?:[^]]+?\|)?(.+?)\]\]", "\\1", line)
        line = re.sub(
            r"\[\[(?:ファイル|Category):([^]|]+)(?:.*?)\]\]", "\\1", line)
        line = re.sub(r"\{\{(?:[^}]+\|)?([^|]+?)\}\}", "\\1", line)
        line = re.sub(r"<.+?>", " ", line)
        line = re.sub(r"\[https?://[^]]+?\]", " ", line)

        if f:
            if re.search(r"^\}\}$", line):
                break
            if re.search(r"^\|(.+?)=(.+)", line):

                m = re.search(r"^\|(.+?)=(.+)", line)
                d_key = m.group(1).strip()
                ans[d_key] = m.group(2).strip()
            else:
                ans[d_key] += line.strip()
    # print(ans["公式国名"])
    return ans


if __name__ == "__main__":
    ans = main()
    for k, v in ans.items():
        print(k, ":", v)

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
        if f:
            if re.search(r"^\}\}$", line):
                break
            if re.search(r"^\|(.+?)=(.+)", line):  # 「|」をエスケープするの忘れてて、うまく動作しなくて焦った。かんそう
                m = re.search(r"^\|(.+?)=(.+)", line)
                d_key = m.group(1).strip()
                ans[d_key] = m.group(2).strip()
            else:
                ans[d_key] += f"\n{line.strip()}"
    return ans


if __name__ == "__main__":
    ans = main()
    from pprint import pprint
    pprint(ans)

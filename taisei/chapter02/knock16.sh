gsplit -n l/3 -d popular-names.txt knock16_check_ #linuxOSとMacではsplitコマンドが異なる? macでは使えるオプションが少ないみたい
#代わりにgsplitで代用 -n 3だと3つのファイルに分割。-n l/3だと3つのファイルに分割する際、行の途中で分割しない。
#https://blog.withachristianwife.com/2021/07/11/split-a-text-fils-by-a-specific-number-of-lines/

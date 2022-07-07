#!/usr/bin/env bash
cut -f 1 popular-names.txt |sort | uniq -c | sort -r -t $'\t' -k 1 -n
#!/usr/bin/env bash
fairseq-interactive --path save91/checkpoint3.pt data91 < test.spacy.ja | grep '^H' | cut -f3 > 92.out
# [ja] dictionary: 49320 types
# [en] dictionary: 61944 types
# Total time: 266.575 seconds; translation time: 253.275


# first 10 lines in file of 92.out
# <unk>
# <unk> ( <unk> ) was a priest of the Rinzai sect in the late Kamakura period .
# He was the founder of the Soto sect .
# He was also known as <unk> .
# He was the founder of the sect .
# His posthumous Buddhist name was <unk> .
# It is also called <unk> .
# It is said to be the origin of the word <unk> or <unk> in Japan .
# It is said that he was a disciple of <unk> .
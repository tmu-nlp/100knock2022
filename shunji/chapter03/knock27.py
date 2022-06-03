from knock25 import result
import re

def remove_markup(text):
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)

    pattern = r'\[\[(?:[^|]*?\|)*?([^|]*?)*?\]\]'
    text = re.sub(pattern, r'\1', text)

    return text

f = open('27.txt', 'w')
result_rm = {k: remove_markup(v) for k, v in result.items()}
for k, v in result_rm.items():
    print(k + ': ' + v)
    f.write(k + ': ' + v + '\n')
f.close()

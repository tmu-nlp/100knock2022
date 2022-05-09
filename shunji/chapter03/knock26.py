from knock25 import result
import re

def remove_markup(text):
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)

    return text

removed = {k: remove_markup(v) for k, v in result.items()}
for k, v in removed.items():
    print(k + ': ' + v)
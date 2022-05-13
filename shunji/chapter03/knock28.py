from knock25 import result
import re

def remove_markup(text):
    pattern = r'\'{2,5}'
    text = re.sub(pattern, '', text)

    pattern = r'\[\[(?:[^|]*?\|)??([^|]*?)\]\]'
    text = re.sub(pattern, r'\1', text)
    
    # ここから追加分
    # htmlタグの削除
    pattern = r'<.+?>'
    text = re.sub(pattern, '', text)

    # テンプレートの削除
    pattern = r'\**\{\{.*?\}\}'
    text = re.sub(pattern, '', text)

    # 外部リンク削除
    pattern = r'\[.*?\]'
    text = re.sub(pattern, '', text)

    return text

f = open('28.txt', 'w')
result_rm = {k: remove_markup(v) for k, v in result.items()}
for k, v in result_rm.items():
    print(k + ': ' + v)
    f.write(k + ': ' + v + '\n') 
f.close()

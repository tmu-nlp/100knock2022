def cipher(text):
    result = ''
    for char in text:
        if char.islower():
            result += chr(219-ord(char))
        else:
            result += char
    return result

cipher('I am Koki Itai')

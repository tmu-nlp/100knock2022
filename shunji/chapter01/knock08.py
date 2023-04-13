def cipher(str):
    l = []
    for fig in str:
        if fig.islower():
            l.append(chr(219-ord(fig)))
        else:
            l.append(fig)
    return ''.join(l)

message = 'The weather tomorrow will be sunny.'

print(cipher(message))
message = cipher(message)
print(cipher(message))
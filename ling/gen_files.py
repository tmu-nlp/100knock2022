import os
chapter=2
for i in range(0,10):
    os.makedirs("chapter0{}".format(chapter),exist_ok=True)
    with open("chapter0{}/knock{}{}.py".format(chapter,chapter-1,i),"w"):
        print("done")
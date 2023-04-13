import matplotlib.pyplot as plt

x = [1, 2, 4, 8, 16]
y = [17.77, 18.61, 19.07, 19.06, 19.46]

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xscale("log",base=2)
plt.savefig("94.png")
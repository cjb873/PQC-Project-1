import numpy as np
import pandas as pd
from inverse import inverse
import time
from plotnine import ggplot, aes, geom_point

longest_time = 0
size = 3
names = ['Dimension', 'Time (s)']

df = pd.DataFrame(columns=names)

while longest_time < 3000.0:
    matrix = np.random.normal(size=[size,size])

    start = time.time()
    inverse(matrix)
    total = time.time() - start

    new_row = pd.DataFrame([dict(zip(names, [size, total]))])
    df = pd.concat([df, new_row], ignore_index=True)
    if total > longest_time:
        longest_time = total
    size+=1

p = (ggplot(df, aes(names[0], names[1])) + geom_point(color="red", size=5))

p.show()

df.to_csv("results.csv")

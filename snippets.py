import pandas as pd

data = pd.DataFrame(columns=['title', 'text'])
data2 = pd.DataFrame({'title': [1, 2, 3], 'text': [2, 4, 6]})
data3 = pd.DataFrame({'title': [4, 5, 6], 'text': [8, 10, 12]})
data4 = data.append(data2, ignore_index=True).append(data3, ignore_index=True)
print(data4)

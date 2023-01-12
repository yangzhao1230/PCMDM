import numpy as np

pth = '5210-0.npy'
data = np.load(pth, allow_pickle=True)

# print(data)
print(type(data))
print(data.item()['motion'].shape)
# print(data['motion'])
print(data.item()['text'])
print(data.item()['lengths'])
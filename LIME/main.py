import numpy as np
import matplotlib.pyplot as plt

# generate the test dataset
# generate the feature a in range

train_x = 10 * np.random.random_sample(size=40)
train_y = np.random.normal((train_x - 5) ** 2, 1)
plt.scatter(train_x, train_y)
plt.show()

test_x = 10 * np.random.random_sample(size=10)
test_y = np.random.normal((train_x - 5) ** 2, 1)

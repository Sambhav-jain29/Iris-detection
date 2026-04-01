from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

df.to_csv("iris_dataset.csv", index=False)

print("Dataset saved as iris_dataset.csv")
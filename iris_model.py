import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
data = pd.read_csv("iris_dataset.csv")

# 2️⃣ Split features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 3️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluate model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 6️⃣ Plot test set predictions
plt.ion()  # Interactive mode
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(
    X_test['petal length (cm)'], 
    X_test['petal width (cm)'], 
    c=y_pred, cmap='viridis', edgecolor='k', s=80, label='Test Set'
)
ax.set_xlabel('Petal Length (cm)')
ax.set_ylabel('Petal Width (cm)')
ax.set_title('Iris Flower Predictions')
ax.legend()
plt.draw()
plt.pause(0.1)

# 7️⃣ Species map for labels
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# 8️⃣ Interactive input loop
while True:
    print("\nEnter new iris measurements (or type 'exit' to quit):")
    try:
        sl = input("Sepal Length: ")
        if sl.lower() == 'exit':
            break
        sw = input("Sepal Width: ")
        if sw.lower() == 'exit':
            break
        pl = input("Petal Length: ")
        if pl.lower() == 'exit':
            break
        pw = input("Petal Width: ")
        if pw.lower() == 'exit':
            break

        # Convert inputs to floats
        sl, sw, pl, pw = float(sl), float(sw), float(pl), float(pw)

        # Convert to DataFrame with correct column names
        flower = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)
        pred = model.predict(flower)[0]
        print("Predicted Species:", species_map[pred])

        # Add new flower to plot
        ax.scatter(pl, pw, c='red', edgecolor='black', s=120, marker='X')
        ax.text(pl+0.02, pw+0.02, species_map[pred], color='red')
        plt.draw()
        plt.pause(0.1)

    except ValueError:
        print("Please enter valid numbers or 'exit' to quit.")

plt.ioff()
plt.show()
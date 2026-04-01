import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# 1️⃣ Load dataset
# ─────────────────────────────────────────────
data = pd.read_csv("/Users/sambhavjain/Downloads/iris_dataset.csv")  # ← update path if needed

# ─────────────────────────────────────────────
# 2️⃣ Split features and labels
# ─────────────────────────────────────────────
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ─────────────────────────────────────────────
# 3️⃣ Train / Val / Test split
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val,   X_test,  y_val,   y_test  = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ─────────────────────────────────────────────
# 4️⃣ Train model with regularization
# ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5️⃣ Accuracy scores
# ─────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc   = accuracy_score(y_val,   model.predict(X_val))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Training Accuracy  : {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}")
print(f"5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

y_pred = model.predict(X_test)

# ─────────────────────────────────────────────
# 📊 PLOT 1 — Correlation Matrix
# ─────────────────────────────────────────────
corr_matrix = data.iloc[:, :-1].corr()

fig1, ax1 = plt.subplots(figsize=(7, 6))
fig1.patch.set_facecolor('#1e1e2e')
ax1.set_facecolor('#1e1e2e')

sns.heatmap(
    corr_matrix,
    ax=ax1,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='#2a2a3e',
    square=True,
    vmin=-1, vmax=1,
    annot_kws={"size": 12, "weight": "bold", "color": "white"},
    cbar_kws={"shrink": 0.8}
)

ax1.set_title('Feature Correlation Matrix', fontsize=15, fontweight='bold', color='white', pad=15)
ax1.tick_params(colors='white', labelsize=10)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha='right', color='white')
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, color='white')

cbar1 = ax1.collections[0].colorbar
cbar1.ax.yaxis.set_tick_params(color='white')
cbar1.outline.set_edgecolor('white')
plt.setp(cbar1.ax.yaxis.get_ticklabels(), color='white')

plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150, bbox_inches='tight', facecolor=fig1.get_facecolor())
plt.show(block=False)

# ─────────────────────────────────────────────
# 📊 PLOT 2 — Confusion Matrix
# ─────────────────────────────────────────────
species_labels = ["Setosa", "Versicolor", "Virginica"]
cm      = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig2, ax2 = plt.subplots(figsize=(7, 6))
fig2.patch.set_facecolor('#1e1e2e')
ax2.set_facecolor('#1e1e2e')

sns.heatmap(
    cm_norm,
    ax=ax2,
    annot=False,
    cmap='Blues',
    linewidths=0.8,
    linecolor='#2a2a3e',
    square=True,
    vmin=0, vmax=1,
    xticklabels=species_labels,
    yticklabels=species_labels,
    cbar_kws={"shrink": 0.8}
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = 'white' if cm_norm[i, j] > 0.5 else '#cccccc'
        ax2.text(j + 0.5, i + 0.5,
                 f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)",
                 ha='center', va='center',
                 fontsize=13, fontweight='bold', color=color)

ax2.set_title(f'Confusion Matrix — Test Set  (Acc: {test_acc*100:.1f}%)',
              fontsize=14, fontweight='bold', color='white', pad=15)
ax2.set_xlabel('Predicted Label', fontsize=12, color='white', labelpad=10)
ax2.set_ylabel('True Label',      fontsize=12, color='white', labelpad=10)
ax2.tick_params(colors='white', labelsize=11)
ax2.set_xticklabels(ax2.get_xticklabels(), color='white')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, color='white')

cbar2 = ax2.collections[0].colorbar
cbar2.ax.yaxis.set_tick_params(color='white')
cbar2.outline.set_edgecolor('white')
plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='white')

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.show(block=False)

# ─────────────────────────────────────────────
# 📊 PLOT 3 — Scatter: Test Set Predictions
# ─────────────────────────────────────────────
species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

fig3, ax3 = plt.subplots(figsize=(7, 5))
fig3.patch.set_facecolor('#1e1e2e')
ax3.set_facecolor('#1e1e2e')

scatter = ax3.scatter(
    X_test['petal length (cm)'],
    X_test['petal width (cm)'],
    c=y_pred, cmap='viridis', edgecolor='white', s=90, linewidths=0.6
)
ax3.set_xlabel('Petal Length (cm)', color='white')
ax3.set_ylabel('Petal Width (cm)',  color='white')
ax3.set_title('Test Set Predictions', fontsize=14, fontweight='bold', color='white')
ax3.tick_params(colors='white')
for spine in ax3.spines.values():
    spine.set_edgecolor('#444')

cbar3 = plt.colorbar(scatter, ax=ax3)
cbar3.set_ticks([0, 1, 2])
cbar3.set_ticklabels(["Setosa", "Versicolor", "Virginica"])
cbar3.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar3.ax.yaxis.get_ticklabels(), color='white')

plt.tight_layout()
plt.show(block=False)

# ─────────────────────────────────────────────
# 🔁 Interactive Input Loop
# ─────────────────────────────────────────────
plt.ion()
while True:
    print("\nEnter new iris measurements (or type 'exit' to quit):")
    try:
        sl = input("Sepal Length: ")
        if sl.lower() == 'exit': break
        sw = input("Sepal Width: ")
        if sw.lower() == 'exit': break
        pl = input("Petal Length: ")
        if pl.lower() == 'exit': break
        pw = input("Petal Width: ")
        if pw.lower() == 'exit': break

        sl, sw, pl, pw = float(sl), float(sw), float(pl), float(pw)
        flower = pd.DataFrame([[sl, sw, pl, pw]], columns=X.columns)

        pred       = model.predict(flower)[0]
        proba      = model.predict_proba(flower)[0]
        confidence = np.max(proba) * 100

        print(f"Predicted Species: {species_map[pred]}  (confidence: {confidence:.1f}%)")

        ax3.scatter(pl, pw, c='red', edgecolor='white', s=130, marker='X', zorder=5)
        ax3.text(pl + 0.05, pw + 0.05,
                 f"{species_map[pred]}\n{confidence:.0f}%",
                 color='red', fontsize=8)
        fig3.canvas.draw()
        plt.pause(0.1)

    except ValueError:
        print("Please enter valid numbers or 'exit' to quit.")

plt.ioff()
plt.show()

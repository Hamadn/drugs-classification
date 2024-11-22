import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Load data
df = pd.read_csv("./drug200.csv")

df.info()
df.head()
df.tail()
df.describe()

# Encode categorical variables
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_cholesterol = LabelEncoder()
le_drug = LabelEncoder()

df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
df["BP_encoded"] = le_bp.fit_transform(df["BP"])
df["Cholesterol_encoded"] = le_cholesterol.fit_transform(df["Cholesterol"])
df["Drug_encoded"] = le_drug.fit_transform(df["Drug"])

# Prepare features and target
X = df[["Age", "Sex_encoded", "BP_encoded", "Cholesterol_encoded", "Na_to_K"]]
y = df["Drug_encoded"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train SVM
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Print results
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_drug.classes_))

# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le_drug.classes_,
    yticklabels=le_drug.classes_,
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = df[
    [
        "Age",
        "Sex_encoded",
        "BP_encoded",
        "Cholesterol_encoded",
        "Na_to_K",
        "Drug_encoded",
    ]
].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

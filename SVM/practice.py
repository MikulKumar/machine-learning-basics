from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
df = datasets.load_iris()

x,y = df.data, df.target

# Create simple scatter plot
plt.figure(figsize=(8, 6))

# Plot each species with different colors
colors = ['red', 'green', 'blue']
for i in range(3):
    mask = y == i
    plt.scatter(x[mask, 0], x[mask, 1], 
               c=colors[i], 
               label=df.target_names[i],
               alpha=0.7)

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

scaled = StandardScaler()
scaled.fit(x_train)
scaled_x_train = scaled.transform(x_train)
scaled_x_test = scaled.transform(x_test)

model = SVC(kernel="rbf", C=1.0, gamma='scale')
model.fit(scaled_x_train,y_train)

y_pred = model.predict(scaled_x_test)
print("accuracy",accuracy_score(y_test,y_pred))

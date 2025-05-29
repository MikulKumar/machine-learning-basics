# Support Vector Machines 
# supervies learning algorithm
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np 

df = datasets.load_iris()
x,y = df.data, df.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=10)


scaler = StandardScaler()
scaler.fit(x_train) # learn the hyperparameters from the training set only

scaled_x_train = scaler.transform(x_train)
scaled_x_test = scaler.transform(x_test)

# kernel : a trick to project data into higher dimensions where it becomes linerally seperable 
'''
| Kernel         | Use Case                                           |
|--------------- | -------------------------------------------------- |
| Linear         | Simple, fast, good when data is linearly separable |
| Polynomial     | Good for curved boundaries                         |
| RBF (Gaussian) | Very popular, flexible for complex shapes          |
| Sigmoid        | Like a neural network activation                   |

'''
model = SVC(kernel="linear", random_state=42)
model.fit(scaled_x_train,y_train)
'''
| Parameter | What it Does                                                                                                                                                |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C`       | Regularization strength. **Low C** → more tolerance to misclassification (smoother decision boundary). **High C** → tries to classify all points correctly. |
| `kernel`  | Specifies the kernel type (`'linear'`, `'poly'`, `'rbf'`, etc.)                                                                                             |
| `gamma`   | Controls the influence of single data points. **High gamma** → model tries to fit the training data very closely.                                           |
'''
accuracy = model.score(scaled_x_test,y_test)
print(f"Accuracy: {accuracy:.5f}")  
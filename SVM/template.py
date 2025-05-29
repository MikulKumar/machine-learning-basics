from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load data
X, y = datasets.load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)
'''
| Parameter | What it Does                                                                                                                                                |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C`       | Regularization strength. **Low C** → more tolerance to misclassification (smoother decision boundary). **High C** → tries to classify all points correctly. |
| `kernel`  | Specifies the kernel type (`'linear'`, `'poly'`, `'rbf'`, etc.)                                                                                             |
| `gamma`   | Controls the influence of single data points. **High gamma** → model tries to fit the training data very closely.                                           |
'''
# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

'''
SVM
├── Goal: Maximize Margin
├── Support Vectors: Closest Points
├── Types
│   ├── Linear SVM
│   └── Kernel SVM
│       ├── RBF
│       ├── Polynomial
│       └── Sigmoid
├── Parameters: C, kernel, gamma
├── Applications: Classification, OCR, Spam detection
└── Libraries: scikit-learn, libsvm
'''
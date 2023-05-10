from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

pred = np.loadtxt("pred.txt")
true = np.loadtxt("y_test.txt")

cm = confusion_matrix(true, pred)
disp = ConfusionMatrixDisplay(cm)
acc_class = cm.diagonal()/cm.sum(axis=1)

plt.bar(x=range(84),height=acc_class)
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.show()
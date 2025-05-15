from qiskit_machine_learning.algorithms import QSVC
from sklearn.metrics import classification_report

def train_qsvc(X_train, y_train, X_test, y_test, kernel):
    model = QSVC(quantum_kernel=kernel)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return model, report
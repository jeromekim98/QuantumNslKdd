from src.data_loader import load_nsl_kdd
from src.quantum_kernel import get_quantum_kernel
from src.model_qsvc import train_qsvc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Load data
df = load_nsl_kdd('data/nsl_kdd.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reduce feature size for quantum (ex: 6 features)
X = X[:, :6]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Quantum kernel
kernel = get_quantum_kernel(num_features=X.shape[1])

# Train model
model, report = train_qsvc(X_train, y_train, X_test, y_test, kernel)

# Save results
with open('results/metrics.json', 'w') as f:
    json.dump(report, f, indent=4)
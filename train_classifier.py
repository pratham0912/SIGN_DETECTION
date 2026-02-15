import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# ----- Load dataset -----
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = []
labels = []

# Keep only valid samples
for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == 42:
        data.append(d)
        labels.append(l)

data = np.asarray(data)
labels = np.asarray(labels)

print("Dataset shape:", data.shape)

# ----- Train-test split -----
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels
)

# ----- BEST MODEL -----
model = SVC(
    kernel='rbf',       # ⭐ nonlinear patterns
    C=10,
    gamma='scale',
    probability=True    # ⭐ for confidence later
)

# ----- Train -----
model.fit(x_train, y_train)

# ----- Evaluate -----
y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)
print(f'Accuracy: {score * 100:.2f}%')

# ----- Save model -----
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as model.p")

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from matplotlib.cbook import get_sample_data

# Завантаження прив'язок символів компаній до їх повних назв
with open('company_symbol_mapping.json', 'r') as f:
    company_mapping = json.load(f)

# Завантаження даних котирувань
datafile = get_sample_data("goog.npz", asfileobj=False)
quotes = np.load(datafile)['price_data']

# Перевірка доступних полів у даних котирувань
print(quotes.dtype.names)

# Витягнення релевантних полів
open_prices = quotes['open']
close_prices = quotes['close']
price_differences = close_prices - open_prices
dates = quotes['date']

# Список символів компаній
company_symbols = list(company_mapping.keys())

# Нормалізація даних
scaler = StandardScaler()
normalized_data = scaler.fit_transform(price_differences.reshape(-1, 1))

# Створення моделі поширення подібності з налаштованими параметрами
affinity_model = AffinityPropagation(random_state=42, max_iter=1000, convergence_iter=50, damping=0.9)
affinity_model.fit(normalized_data)

# Отримання результатів кластеризації
cluster_centers_indices = affinity_model.cluster_centers_indices_
labels = affinity_model.labels_

# Виведення результатів
clusters = {}
for idx, label in enumerate(labels):
    company_symbol = company_symbols[idx % len(company_symbols)]
    company_name = company_mapping.get(company_symbol, "Unknown")
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(company_name)

print("Результати кластеризації:")
for cluster_id, members in clusters.items():
    print(f"Кластер {cluster_id}: {', '.join(members)}")

# Візуалізація кластерів
plt.scatter(normalized_data, np.zeros_like(normalized_data), c=labels, cmap='viridis')
plt.title("Кластеризація на основі поширення подібності")
plt.xlabel("Нормалізовані різниці котирувань")
plt.ylabel("Кластери")
plt.show()

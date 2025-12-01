import matplotlib.pyplot as plt

threshlod = [3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9]
detections = [92, 83, 85, 83, 83, 80, 80]

# Create scatterplot and connect points
plt.figure(figsize=(6, 4))
plt.scatter(threshlod, detections, color='red', label='Data points')   # scatter points
plt.plot(threshlod, detections, color='blue', label='Connected line')  # connect the dots

plt.xlabel('threshlod')
plt.ylabel('detections')
plt.title('detections vs threshlod')
plt.legend()
# plt.grid(True)
plt.savefig('detections vs threshlod')
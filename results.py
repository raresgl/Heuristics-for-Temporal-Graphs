import matplotlib.pyplot as plt

# Data points
T_values = [10000, 30000, 40000, 50000]
total_time_values = [21, 121.14, 532.64, 1363.57]
precision = [0.99460520718889, 0.9936007823581818, 0.9955412049925715,0.9931331791710662]
precision_percentage = [p * 100 for p in precision]
# Create the plot
plt.figure(figsize=(8, 5))

plt.plot(T_values, precision_percentage, marker='o', linestyle='-', color='b', label="Execution Time")
# Labels and title
plt.xlabel("Total timestamp values")
plt.ylabel("Precision")
plt.title("Precision vs T")
plt.ylim(0, 100)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#plt.plot(T_values, total_time_values, marker='o', linestyle='-', color='b', label="Execution Time")
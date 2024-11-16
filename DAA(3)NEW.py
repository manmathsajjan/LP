# Class to represent an item in the knapsack
class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight

    # Function to calculate value-to-weight ratio
    def value_per_weight(self):
        return self.value / self.weight

# Function to solve the Fractional Knapsack Problem
def fractional_knapsack(items, capacity):
    # Sort items by value-to-weight ratio in descending order
    items.sort(key=lambda item: item.value_per_weight(), reverse=True)

    total_value = 0.0  # Total value accumulated in the knapsack
    total_weight = 0   # Total weight accumulated in the knapsack

    # Loop through the sorted items
    for item in items:
        if total_weight + item.weight <= capacity:
            # If adding the whole item doesn't exceed capacity, take the whole item
            total_weight += item.weight
            total_value += item.value
        else:
            # Otherwise, take a fraction of the item to fill the knapsack
            remaining_capacity = capacity - total_weight
            fraction = remaining_capacity / item.weight
            total_value += item.value * fraction
            total_weight += item.weight * fraction
            break  # Knapsack is now full

    return total_value

# Main function to test the Fractional Knapsack
if __name__ == "__main__":
    # Taking input from the user
    n = int(input("Enter the number of items: "))

    items = []
    for i in range(n):
        value = float(input(f"Enter value of item {i + 1}: "))
        weight = float(input(f"Enter weight of item {i + 1}: "))
        items.append(Item(value, weight))

    # Input: Capacity of the knapsack
    capacity = float(input("Enter the capacity of the knapsack: "))

    # Calculate the maximum value of the knapsack
    max_value = fractional_knapsack(items, capacity)

    print(f"\nMaximum value in Knapsack: {max_value:.2f}")

def knapsack_dp(W, wt, val, n):
    """A Dynamic Programming based solution for 0-1 Knapsack problem
    Returns the maximum value that can be put in a knapsack of capacity W."""
    
    K = [[0 for x in range(W + 1)] for x in range(n + 1)]

    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    
    return K[n][W]

# Taking input from the user
n = int(input("Enter the number of items: "))
val = []
wt = []

for i in range(n):
    value = int(input(f"Enter value of item {i + 1}: "))
    weight = int(input(f"Enter weight of item {i + 1}: "))
    val.append(value)
    wt.append(weight)

W = int(input("Enter the capacity of the knapsack: "))

# Calculate and print the maximum possible profit
print("Maximum possible profit =", knapsack_dp(W, wt, val, n))

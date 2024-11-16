# Iterative Fibonacci function
def fibonacci_iterative(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b

# Recursive Fibonacci function
def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Memoized Fibonacci function
def fibonacci_memoization(n, memo={}):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    if n not in memo:
        memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
    return memo[n]

# Taking user input
try:
    n = int(input("Enter a number to calculate the Fibonacci: "))
    
    # Ensure the input is non-negative
    if n < 0:
        print("Please enter a non-negative number.")
    else:
        # Iterative Fibonacci
        result_iter = fibonacci_iterative(n)
        print(f"Fibonacci Iterative of {n} is: {result_iter}")
        print(f"Time Complexity: O(n)")
        print(f"Space Complexity: O(1)\n")
        
        # Recursive Fibonacci
        result_rec = fibonacci_recursive(n)
        print(f"Fibonacci Recursive of {n} is: {result_rec}")
        print(f"Time Complexity: O(2^n)")
        print(f"Space Complexity: O(n) (due to recursive stack)\n")
        
        # Memoized Fibonacci
        result_memo = fibonacci_memoization(n)
        print(f"Fibonacci Memoization of {n} is: {result_memo}")
        print(f"Time Complexity: O(n)")
        print(f"Space Complexity: O(n) (due to memo dictionary)\n")

except ValueError:
    print("Invalid input! Please enter an integer.")

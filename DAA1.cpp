#include <iostream>
using namespace std;

int fibonacciIterative(int n) {
    if (n <= 1)
        return n;
    int a = 0, b = 1, fib;
    for (int i = 2; i <= n; ++i) {
        fib = a + b;
        a = b;
        b = fib;
    }
    return fib;
}

int fibonacciRecursive(int n) {
    if (n <= 1)
        return n;
    return fibonacciRecursive(n - 1) + fibonacciRecursive(n - 2);
}

int main() {
    int n;
    cout << "Enter a number to calculate Fibonacci: ";
    cin >> n;

    cout << "Fibonacci (Iterative) of " << n << " is: " << fibonacciIterative(n) << endl;
    cout << "Fibonacci (Recursive) of " << n << " is: " << fibonacciRecursive(n) << endl;

    return 0;
}

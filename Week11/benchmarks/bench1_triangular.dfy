// Benchmark 1: Triangular Numbers
// Sum of 0 + 1 + 2 + ... + (i-1) = i*(i-1)/2
// Quadratic invariant: 2*sum == i*i - i (or i*i - i - 2*sum == 0)

method TriangularSum(n: int) returns (sum: int)
  requires n >= 0
  ensures sum == n * (n - 1) / 2
{
  var i := 0;
  sum := 0;
  
  while i < n
    // Expected quadratic invariant:
    // 2 * sum == i * i - i
    // Equivalently: i * i - i - 2 * sum == 0
    invariant 0 <= i <= n
    invariant 2 * sum == i * (i - 1)
  {
    sum := sum + i;
    i := i + 1;
  }
}

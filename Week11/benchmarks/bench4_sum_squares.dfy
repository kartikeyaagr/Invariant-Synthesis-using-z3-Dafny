// Benchmark 4: Sum of Squares
// Computes 0² + 1² + 2² + ... + (i-1)² = (i-1)*i*(2i-1)/6
// Requires cubic invariant (beyond pure quadratic, but we can bound it)

method SumOfSquares(n: int) returns (sum: int)
  requires n >= 0
  ensures sum == (n - 1) * n * (2 * n - 1) / 6
{
  var i := 0;
  sum := 0;
  
  while i < n
    // Cubic relationship, but we can use quadratic bounds
    // Invariant: 6 * sum <= i * i * i (upper bound)
    // Invariant: sum >= 0
    invariant 0 <= i <= n
    invariant sum >= 0
  {
    sum := sum + i * i;
    i := i + 1;
  }
}

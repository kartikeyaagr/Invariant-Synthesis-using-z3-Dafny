// Benchmark 2: Square Computation
// Computes n² using repeated addition
// Quadratic invariant: sq == i * i

method ComputeSquare(n: int) returns (sq: int)
  requires n >= 0
  ensures sq == n * n
{
  var i := 0;
  sq := 0;
  var odd := 1;
  
  while i < n
    // Expected: sq == i * i
    // Using odd numbers: 1 + 3 + 5 + ... + (2i-1) = i²
    invariant 0 <= i <= n
    invariant sq == i * i
    invariant odd == 2 * i + 1
  {
    sq := sq + odd;
    odd := odd + 2;
    i := i + 1;
  }
}

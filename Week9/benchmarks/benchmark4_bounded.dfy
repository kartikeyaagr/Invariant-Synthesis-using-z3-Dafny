// Benchmark 4: Bounded Counter
// Requires conjunction of bounds: (i >= 0) && (i <= n) && (sum >= 0)

method BoundedCounter(n: int) returns (sum: int)
  requires n >= 0
  ensures sum == n * (n + 1) / 2
{
  var i := 0;
  sum := 0;
  
  while i <= n
    // Invariants to be synthesized:
    // invariant 0 <= i
    // invariant i <= n + 1
    // invariant sum >= 0
  {
    sum := sum + i;
    i := i + 1;
  }
}

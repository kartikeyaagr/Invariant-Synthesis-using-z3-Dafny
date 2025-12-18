// Benchmark 5: Conditional Accumulation
// Accumulate only when condition is met
// Requires: disjunctive invariant for accumulator bounds

method ConditionalAccumulator(n: int) returns (sum: int)
  requires n >= 0
  ensures sum >= 0
{
  var i := 0;
  sum := 0;
  
  while i < n
    // When i is even, we add to sum
    // When i is odd, we skip
    // Invariant: sum counts even numbers seen
  {
    if i % 2 == 0 {
      sum := sum + i;
    }
    // else: no update to sum
    i := i + 1;
  }
}

// Benchmark 3: Positive/Negative Value Check
// Update based on sign of a value
// Requires: (x > 0 && inv1) || (x <= 0 && inv2)

method SignBasedUpdate(n: int) returns (count: int, sum: int)
  requires n >= 0
  ensures count >= 0
{
  var i := 0;
  var x := n;
  count := 0;
  sum := 0;
  
  while i < n
    // Expected:
    // (x > 0 ==> count increases) && (x <= 0 ==> sum increases)
  {
    if x > 0 {
      count := count + 1;
      x := x - 1;
    } else {
      sum := sum + 1;
    }
    i := i + 1;
  }
}

// Benchmark 4: Three-Way Branch
// Multiple disjoint conditions with different updates
// Requires: (cond1 && inv1) || (cond2 && inv2) || (else && inv3)

method ThreeWayBranch(n: int) returns (a: int, b: int, c: int)
  requires n >= 0
  ensures a >= 0 && b >= 0 && c >= 0
{
  var i := 0;
  a := 0;
  b := 0;
  c := 0;
  
  while i < n
    // Expected disjunctive invariant for three paths:
    // (i % 3 == 0 ==> a updates) && 
    // (i % 3 == 1 ==> b updates) && 
    // (i % 3 == 2 ==> c updates)
  {
    if i % 3 == 0 {
      a := a + 1;
    } else {
      if i % 3 == 1 {
        b := b + 1;
      } else {
        c := c + 1;
      }
    }
    i := i + 1;
  }
}

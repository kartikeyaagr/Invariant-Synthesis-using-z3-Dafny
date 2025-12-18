// Benchmark 3: Three-variable Update
// Requires: (x >= 0) && (y >= 0) && (z >= 0) && (x + y + z == 3*i)

method ThreeVarUpdate(n: int) returns (x: int, y: int, z: int)
  requires n >= 0
  ensures x == n
  ensures y == n
  ensures z == n
{
  x := 0;
  y := 0;
  z := 0;
  
  while x < n
    // Invariants to be synthesized:
    // invariant x >= 0 && y >= 0 && z >= 0
    // invariant x == y && y == z
    // invariant x <= n
  {
    x := x + 1;
    y := y + 1;
    z := z + 1;
  }
}

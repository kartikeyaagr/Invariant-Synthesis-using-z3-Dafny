// Benchmark 3: Product via Addition
// Computes a * b using repeated addition
// Invariant involves product: prod == i * b (linear in this case)

method MultiplyByAddition(a: int, b: int) returns (prod: int)
  requires a >= 0
  requires b >= 0
  ensures prod == a * b
{
  var i := 0;
  prod := 0;
  
  while i < a
    // Expected: prod == i * b
    invariant 0 <= i <= a
    invariant prod == i * b
  {
    prod := prod + b;
    i := i + 1;
  }
}

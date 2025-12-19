// Benchmark 7: Count Occurrences
// Count how many times a value appears
// Invariant: count equals occurrences in a[0..i]

function CountIn(a: array<int>, val: int, n: int): int
  requires 0 <= n <= a.Length
  reads a
{
  if n == 0 then 0 
  else if a[n-1] == val then CountIn(a, val, n-1) + 1
  else CountIn(a, val, n-1)
}

method CountOccurrences(a: array<int>, val: int) returns (count: int)
  ensures count == CountIn(a, val, a.Length)
{
  var i := 0;
  count := 0;
  
  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant count == CountIn(a, val, i)
  {
    if a[i] == val {
      count := count + 1;
    }
    i := i + 1;
  }
}

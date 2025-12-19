// Benchmark 2: Array Sum
// Compute sum of all array elements
// Invariant: sum equals sum of a[0..i]

function SumTo(a: array<int>, n: int): int
  requires 0 <= n <= a.Length
  reads a
{
  if n == 0 then 0 else SumTo(a, n-1) + a[n-1]
}

method ArraySum(a: array<int>) returns (sum: int)
  ensures sum == SumTo(a, a.Length)
{
  var i := 0;
  sum := 0;
  
  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant sum == SumTo(a, i)
  {
    sum := sum + a[i];
    i := i + 1;
  }
}

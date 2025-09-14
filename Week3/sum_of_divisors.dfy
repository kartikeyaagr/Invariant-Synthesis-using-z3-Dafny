function SumDivUpTo(n: int, D: int): int
  requires n >= 0 && 0 <= D <= n
  decreases D
{
  if D == 0 then 0
  else SumDivUpTo(n, D - 1) + (if n % D == 0 then D else 0)
}

function SumOfDivisors(n: int): int
  requires n >= 0
{
  SumDivUpTo(n, n)
}

method ComputeSumOfDivisors(n: int) returns (s: int)
  requires n >= 0
  ensures s == SumOfDivisors(n)
{
  s := 0;
  var d := 1;
  while d <= n
    invariant 1 <= d <= n + 1
    invariant s == SumDivUpTo(n, d - 1)
    decreases n - d + 1
  {
    if n % d == 0 {
      s := s + d;
    }
    d := d + 1;
  }
}

method Main() {
  var n := 12;
  var s := ComputeSumOfDivisors(n);
  print "Sum of divisors of ", n, " is ", s, "\n";
}

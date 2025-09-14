function Factorial(n: nat): nat
  decreases n
{
  if n == 0 then 1 else n * Factorial(n - 1)
}

method ComputeFactorial(n: nat) returns (f: nat)
  requires n >= 0
  ensures f == Factorial(n)
{
  f := 1;
  var i := 0;
  while i < n
    invariant 0 <= i <= n
    invariant f == Factorial(i)
    decreases n - i
  {
    i := i + 1;
    f := f * i;
  }
}

method Main() {
  var n := 6;
  var f := ComputeFactorial(n);
  print n, "! = ", f, "\n";
}

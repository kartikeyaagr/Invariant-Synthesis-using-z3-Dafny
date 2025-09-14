function Triangular(n: nat): nat
  decreases n
{
  if n == 0 then 0 else n + Triangular(n - 1)
}

method ComputeTriangular(n: nat) returns (t: nat)
  ensures t == Triangular(n)
{
  t := 0;
  var i := 1;
  while i <= n
    invariant 1 <= i <= n + 1
    invariant t == Triangular(i - 1)
    decreases n - i + 1
  {
    t := t + i;
    i := i + 1;
  }
}

method Main() {
  var n := 10;
  var t := ComputeTriangular(n);
  print "T(", n, ") = ", t, "\n";
}

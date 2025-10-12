method Sum(n: int) returns (result: int)
  requires n >= 0
{
  var i := 0;
  var sum := 0;
  
  while (i <= n)
  {
    sum := sum + i;
    i := i + 1;
  }
  
  result := sum;
}
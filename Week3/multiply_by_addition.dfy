method MultiplyByAddition(a: nat, b: nat) returns (product: nat)
  ensures product == a * b
{
  product := 0;
  var i := 0;
  while i < b
    invariant 0 <= i <= b
    invariant product == a * i
    decreases b - i
  {
    product := product + a;
    i := i + 1;
  }
}

method Main() {
  var a := 7;
  var b := 5;
  var p := MultiplyByAddition(a, b);
  print a, " * ", b, " = ", p, "\n";
}

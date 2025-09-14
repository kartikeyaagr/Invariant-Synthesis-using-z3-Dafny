function Power(x:int, n:nat):int
decreases n
{
if n == 0 then 1 else x * Power(x, n-1)
}

lemma Power_add(x:int, m:nat, n:nat)
ensures Power(x, m + n) == Power(x, m) * Power(x, n)
decreases m
{
if m == 0 { } else {
    Power_add(x, m-1, n);
    assert Power(x, m + n) == x * Power(x, m-1 + n);
    assert Power(x, m) == x * Power(x, m-1);
    assert Power(x, m-1 + n) == Power(x, m-1) * Power(x, n);
    assert Power(x, m + n) == Power(x, m) * Power(x, n);
}
}

lemma Power_xx_eq_square(x:int, n:nat)
ensures Power(x*x, n) == Power(x, n) * Power(x, n)
decreases n
{
if n == 0 { } else {
    Power_xx_eq_square(x, n-1);
    assert Power(x*x, n) == (x*x) * Power(x*x, n-1);
    assert Power(x, n) == x * Power(x, n-1);
    assert (x*x) * Power(x*x, n-1) == (x * Power(x, n-1)) * (x * Power(x, n-1));
    assert Power(x*x, n) == Power(x, n) * Power(x, n);
}
}

lemma Power_even(x:int, q:nat)
ensures Power(x, 2*q) == Power(x*x, q)
decreases q
{
Power_add(x, q, q);
Power_xx_eq_square(x, q);
assert Power(x, 2*q) == Power(x, q) * Power(x, q);
assert Power(x, q) * Power(x, q) == Power(x*x, q);
assert Power(x, 2*q) == Power(x*x, q);
}


lemma Power_double_odd(x:int, q:nat)
ensures Power(x, 2*q + 1) == x * Power(x*x, q)
decreases q
{
Power_add(x, 2*q, 1);      
Power_even(x, q);           
assert Power(x, 1) == x;
assert Power(x, 2*q + 1) == x * Power(x*x, q);
}

method FastPower(base:int, exp:nat) returns (result:int)
ensures result == Power(base, exp)
{
var x := base;
var n := exp;
result := 1;

while n > 0
    invariant n >= 0
    invariant result * Power(x, n) == Power(base, exp)
    decreases n
{
    if n % 2 == 1 {
    var q := n / 2;
    assert n == 2*q + 1;
    Power_double_odd(x, q);

    result := result * x;
    x := x * x;
    n := q;
    } else {
    var q := n / 2;
    assert n == 2*q;
    Power_even(x, q);

    x := x * x;
    n := q;
    }
}
}
method Main() {
  var b := 3;
  var e := 5;
  var r := FastPower(b, e);
  print b, "^", e, " = ", r, "\n";
}

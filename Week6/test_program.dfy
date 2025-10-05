method test(n: int) returns (x: int, y: int)
    requires n >= 0
{
    x := 0;
    y := 0;
    while (x < n)
    {
        x := x + 1;
        y := y + 2;
    }
}
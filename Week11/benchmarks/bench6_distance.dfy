// Benchmark 6: Coordinate Distance
// Moving along a line, tracking squared distance
// Invariant: dist_sq == x*x + y*y

method CoordinateMove(n: int) returns (x: int, y: int, dist_sq: int)
  requires n >= 0
  ensures x == n
  ensures y == 2 * n
  ensures dist_sq == x * x + y * y
{
  x := 0;
  y := 0;
  dist_sq := 0;
  
  while x < n
    // Track squared distance: dist_sq = x² + y²
    // Since y = 2x, dist_sq = x² + 4x² = 5x²
    invariant 0 <= x <= n
    invariant y == 2 * x
    invariant dist_sq == x * x + y * y
  {
    // Update distance: new_dist = (x+1)² + (y+2)²
    //                          = x² + 2x + 1 + y² + 4y + 4
    //                          = dist_sq + 2x + 1 + 4y + 4
    //                          = dist_sq + 2x + 4y + 5
    dist_sq := dist_sq + 2 * x + 4 * y + 5;
    x := x + 1;
    y := y + 2;
  }
}

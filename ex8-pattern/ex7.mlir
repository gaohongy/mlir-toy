func.func @test(%a: i32, %b: i32) -> i32 {
  %c = "toy.add"(%a, %b): (i32, i32) -> i32
  %d = "toy.add"(%a, %b, %c): (i32, i32, i32) -> i32
  "toy.return"(%c) : (i32) -> ()
}
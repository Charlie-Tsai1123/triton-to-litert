module {
  func.func @main(
      %a: tensor<1024xf32>,
      %b: tensor<1024xf32>) -> tensor<1024xf32> {

    %empty = tensor.empty() : tensor<1024xf32>

    %0 = linalg.elementwise
      kind=#linalg.elementwise_kind<add>
      ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
      outs(%empty : tensor<1024xf32>)
      -> tensor<1024xf32>

    return %0 : tensor<1024xf32>
  }
}

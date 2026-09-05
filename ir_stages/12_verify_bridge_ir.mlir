module attributes {triton_to_litert.bridge_version = 1 : i32, triton_to_litert.entry_point = "vector_add"} {
  func.func @vector_add(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    return %1 : tensor<1024xf32>
  }
}


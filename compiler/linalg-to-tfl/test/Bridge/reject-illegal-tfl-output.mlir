// RUN: linalg-to-tfl-opt --verify-triton-to-litert-tfl-output --verify-diagnostics --split-input-file %s

module {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'tensor.empty': only tfl.add is supported}}
    %empty = tensor.empty() : tensor<1024xf32>
    return %empty : tensor<1024xf32>
  }
}

// -----

module {
  // expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'func.func': body must contain only tfl.add and return}}
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

module {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'tfl.mul': only tfl.add is supported}}
    %result = "tfl.mul"(%a, %b) {fused_activation_function = "NONE"}
        : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'builtin.module': exactly one entry function is required}}
module {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %result = "tfl.add"(%a, %b) {fused_activation_function = "NONE"}
        : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
  func.func private @extra()
}

// -----

module {
  // expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'func.func': output function must retain the milestone ABI}}
  func.func @vector_add(%a: tensor<512xf32>, %b: tensor<512xf32>)
      -> tensor<512xf32> {
    %result = "tfl.add"(%a, %b) {fused_activation_function = "NONE"}
        : (tensor<512xf32>, tensor<512xf32>) -> tensor<512xf32>
    return %result : tensor<512xf32>
  }
}

// -----

module {
  // expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'func.func': output function must retain the milestone ABI}}
  func.func @vector_add(%a: tensor<1024xf16>, %b: tensor<1024xf16>)
      -> tensor<1024xf16> {
    %result = "tfl.add"(%a, %b) {fused_activation_function = "NONE"}
        : (tensor<1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>
    return %result : tensor<1024xf16>
  }
}

// -----

module {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{Triton-to-LiteRT illegal TFL output construct 'tfl.add': fused activation must be NONE}}
    %result = "tfl.add"(%a, %b) {fused_activation_function = "RELU"}
        : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

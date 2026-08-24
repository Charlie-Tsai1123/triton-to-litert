// RUN: linalg-to-tfl-opt --triton-to-litert-bridge-to-tfl --verify-diagnostics --split-input-file %s

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT illegal Bridge IR v1 construct 'linalg.elementwise': linalg.elementwise kind must be add}}
    %result = linalg.elementwise kind=#linalg.elementwise_kind<mul>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  // expected-error @+1 {{Triton-to-LiteRT illegal Bridge IR v1 construct 'func.func': both inputs must be tensor<1024xf32>}}
  func.func @vector_add(%a: tensor<512xf32>, %b: tensor<512xf32>)
      -> tensor<512xf32> {
    %empty = tensor.empty() : tensor<512xf32>
    %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<512xf32>, tensor<512xf32>)
        outs(%empty : tensor<512xf32>) -> tensor<512xf32>
    return %result : tensor<512xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  // expected-error @+1 {{Triton-to-LiteRT illegal Bridge IR v1 construct 'func.func': both inputs must be tensor<1024xf32>}}
  func.func @vector_add(%a: tensor<1024xf16>, %b: tensor<1024xf16>)
      -> tensor<1024xf16> {
    %empty = tensor.empty() : tensor<1024xf16>
    %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf16>, tensor<1024xf16>)
        outs(%empty : tensor<1024xf16>) -> tensor<1024xf16>
    return %result : tensor<1024xf16>
  }
}

// -----

// expected-error @+1 {{Triton-to-LiteRT illegal Bridge IR v1 construct 'builtin.module': module must contain exactly one function}}
module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
  func.func private @extra()
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT illegal Bridge IR v1 construct 'linalg.generic': expected linalg.elementwise}}
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %out: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
    } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{Triton-to-LiteRT illegal Bridge IR v1 construct 'builtin.module': module has unsupported attributes}}
module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add",
  test.unsupported = true
} {
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

// expected-error @+1 {{Triton-to-LiteRT unsupported Bridge IR version: expected i32 version 1}}
module attributes {
  triton_to_litert.bridge_version = 2 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
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

// expected-error @+1 {{Triton-to-LiteRT invalid Bridge IR entry point: expected 'vector_add'}}
module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "wrong_entry"
} {
  func.func @wrong_entry(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

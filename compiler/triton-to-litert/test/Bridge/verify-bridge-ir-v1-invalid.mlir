// RUN: triton-to-litert-opt --verify-triton-to-litert-bridge-ir-v1 --verify-each --verify-diagnostics --allow-unregistered-dialect --split-input-file %s

// expected-error @+1 {{unsupported Bridge IR version: expected i32 version 1}}
module attributes {triton_to_litert.entry_point = "vector_add"} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{unsupported Bridge IR version: expected i32 version 1}}
module attributes {
  triton_to_litert.bridge_version = 2 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{invalid Bridge IR entry point}}
module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "missing"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{invalid Bridge IR entry point}}
module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
  func.func private @extra()
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  // expected-error @+1 {{illegal Bridge IR v1 construct 'func.func': entry signature must be (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32> with one block}}
  func.func @vector_add(%a: tensor<?xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    return %b : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  // expected-error @+1 {{illegal Bridge IR v1 construct 'func.func': entry signature must be (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32> with one block}}
  func.func @vector_add(%a: !tt.ptr<f32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    return %b : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{illegal Bridge IR v1 construct 'linalg.generic': operation is outside the closed Bridge IR v1 allowlist}}
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %init: f32):
        %added = arith.addf %lhs, %rhs : f32
        linalg.yield %added : f32
    } -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{illegal Bridge IR v1 construct 'bufferization.alloc_tensor': operation is outside the closed Bridge IR v1 allowlist}}
    %allocated = bufferization.alloc_tensor() : tensor<1024xf32>
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{illegal Bridge IR v1 construct 'arith.constant': operation is outside the closed Bridge IR v1 allowlist}}
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{illegal Bridge IR v1 construct 'linalg.elementwise': expected one same-shaped binary f32 add with one tensor result}}
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<mul>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{illegal Bridge IR v1 construct 'linalg.elementwise': operands must preserve ABI input order and use tensor.empty as the destination}}
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%b, %a : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{illegal Bridge IR v1 construct 'example.unversioned': module attributes are closed to version and entry point metadata}}
module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add",
  example.unversioned
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

// expected-error @+1 {{invalid Bridge IR entry point}}
module attributes {triton_to_litert.bridge_version = 1 : i32} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  // expected-error @+1 {{illegal Bridge IR v1 construct 'func.func': entry signature must be (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32> with one block}}
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
                        %out: !tt.ptr<f32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %out_ptr = tts.make_tptr %out to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%out_ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{illegal Bridge IR v1 construct 'memref.alloc': operation is outside the closed Bridge IR v1 allowlist}}
    %memory = memref.alloc() : memref<1024xf32>
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{illegal Bridge IR v1 construct 'test.side_effect': operation is outside the closed Bridge IR v1 allowlist}}
    "test.side_effect"() : () -> ()
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// -----

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    // expected-error @+1 {{illegal Bridge IR v1 construct 'builtin.unrealized_conversion_cast': operation is outside the closed Bridge IR v1 allowlist}}
    %cast = builtin.unrealized_conversion_cast %a
        : tensor<1024xf32> to tensor<1024xf32>
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%cast, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// RUN: triton-to-litert-opt --functionalize-triton-to-litert-structured-output --verify-each --verify-diagnostics --split-input-file %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @masked_store(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64: 1024>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @malformed_classification(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{illegal Bridge IR v1 construct 'linalg.elementwise': operands must preserve ABI input order}}
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%b, %a : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_stored_value(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{stored value must be the classified elementwise add result}}
    "tts.store"(%ptr, %a) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @pointer_escape(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    // expected-error @+1 {{output pointer must have one store interpretation and must not escape or be read}}
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %escaped = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "output", "output"],
  triton_to_litert.buffers_distinct
} {
  // expected-error @+1 {{expected distinct buffer roles [input, input, output]}}
  func.func @conflicting_roles(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @partial_store(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    // expected-error @+1 {{unsupported structured pointer layout: size must be static extent 1024}}
    %ptr = tts.make_tptr %out to sizes: [512], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @nonzero_offset(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    // expected-error @+1 {{unsupported structured pointer layout: offset must be static zero after launch normalization}}
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [1], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @strided_store(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    // expected-error @+1 {{unsupported structured pointer layout: stride must be static positive unit stride}}
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [2],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @overlapping_stores(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured store is not a single full-tensor output: expected exactly one tts.store: found 2}}
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @output_read_modify_write(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{model ABI cannot be functionalized: declared write-only output buffer is read, creating unsupported read-modify-write state}}
    %old = "tts.load"(%ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%old, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  // expected-error @+1 {{expected two tensor<1024xf32> inputs and one !tt.ptr<f32> output}}
  func.func @input_targeting_store(
      %a: !tt.ptr<f32>, %b: tensor<1024xf32>, %out: tensor<1024xf32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%empty, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @post_store_operation(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{the sole output store must be immediately followed by the void return with no observable operation after it}}
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    %after = tensor.empty() : tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @boundary_store(%a: tensor<1024xf32>, %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        {boundary_check}
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

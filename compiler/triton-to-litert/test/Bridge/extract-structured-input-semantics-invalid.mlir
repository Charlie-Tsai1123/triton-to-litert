// RUN: triton-to-litert-opt --extract-triton-to-litert-structured-input-semantics --verify-each --verify-diagnostics --split-input-file %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @nonzero_offset(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                            %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: offset must be static zero after launch normalization}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [1],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @boundary_check(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                            %out: !tt.ptr<f32>) {
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor: unsupported load semantic attribute 'boundary_check'}}
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}> {boundary_check}
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_load_rank(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                             %out: !tt.ptr<f32>) {
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{rank-1 f32 tensors must have static extent 1024}}
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1x1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_load_extent(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                               %out: !tt.ptr<f32>) {
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{rank-1 f32 tensors must have static extent 1024}}
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<512xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_load_element_type(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                                     %out: !tt.ptr<f32>) {
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{rank-1 f32 tensors must have static extent 1024}}
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf16>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @residual_normalized_operation(%a: !tt.ptr<f32>,
                                           %b: !tt.ptr<f32>,
                                           %out: !tt.ptr<f32>) {
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    // expected-error @+1 {{normalized bridge preparation does not support operation 'arith.constant'}}
    %residual = arith.constant 0 : i32
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "output", "input"],
  triton_to_litert.buffers_distinct
} {
  // expected-error @+1 {{model ABI cannot be functionalized: expected distinct buffer roles}}
  func.func @wrong_input_roles(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                               %out: !tt.ptr<f32>) {
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @dynamic_offset(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                           %out: !tt.ptr<f32>) {
    %zero = arith.constant 0 : index
    // expected-error @+1 {{unsupported structured pointer layout: offset must be static zero after launch normalization}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%zero], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @nonunit_stride(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                           %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: stride must be static positive unit stride}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [2], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @broadcast_stride(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                              %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: stride must be static positive unit stride}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [0], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_size(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                        %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: size must be static extent 1024}}
    %a_ptr = tts.make_tptr %a to sizes: [512], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_rank(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                        %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: result must be rank-1 tensor<1024x!tt.ptr<f32>>}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1x1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1x1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrong_element_type(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                                %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: element type must be f32}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f16>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f16>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @wrapping_shape(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                            %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: shape must disable wraparound and broadcasting}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [1024], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @reordered(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                       %out: !tt.ptr<f32>) {
    // expected-error @+1 {{unsupported structured pointer layout: order must be empty}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: [0]
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @masked(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                    %out: !tt.ptr<f32>) {
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64: 1024>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @with_other(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                        %out: !tt.ptr<f32>) {
    %other = arith.constant 0.0 : f32
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    %a_value = "tts.load"(%a_ptr, %other)
        <{operandSegmentSizes = array<i32: 1, 0, 1>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, f32) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @pointer_escape(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                            %out: !tt.ptr<f32>) {
    // expected-error @+1 {{input pointer must have exactly one load interpretation and must not escape}}
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %first = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %second = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @output_load(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>,
                         %out: !tt.ptr<f32>) {
    %out_ptr = tts.make_tptr %out to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{the write-only output buffer cannot be used as an input load}}
    %out_value = "tts.load"(%out_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1], offsets: [0],
        shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    return
  }
}

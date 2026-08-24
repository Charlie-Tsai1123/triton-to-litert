// RUN: triton-to-litert-opt --verify-triton-to-litert-bridge-input --verify-each --verify-diagnostics --split-input-file %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @masked_load(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    %mask_extent = arith.constant 512 : index
    %ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    %unused = "tts.load"(%ptr, %mask_extent)
        <{operandSegmentSizes = array<i32: 1, 1, 0>,
          static_mask_dims = array<i64: -9223372036854775808>}>
        : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @zero_stride(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: stride must be static positive unit stride}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [0],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @negative_stride(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: stride must be static positive unit stride}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [-1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @masked_store(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    %mask_extent = arith.constant 512 : index
    %value = arith.constant dense<0.0> : tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    "tts.store"(%ptr, %value, %mask_extent)
        <{static_mask_dims = array<i64: -9223372036854775808>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @static_nonzero_offset(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    // expected-error @+1 {{unsupported structured pointer layout: offset must be the canonical program_id.x tile expression}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [1], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @load_with_other(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    %other = arith.constant 0.0 : f32
    %ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{structured access must be unmasked and cover the full tensor}}
    %unused = "tts.load"(%ptr, %other)
        <{operandSegmentSizes = array<i32: 1, 0, 1>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, f32) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @dynamic_offset(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %unsupported = arith.index_cast %num_x : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: offset must be the canonical program_id.x tile expression}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%unsupported], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @non_unit_stride(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: stride must be static positive unit stride}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [2],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @wrong_rank(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: result must be rank-1 tensor<1024x!tt.ptr<f32>>}}
    %unused = tts.make_tptr %a to sizes: [32, 32], strides: [32, 1],
        offsets: [%tile, 0], shape: [0, 0], order: []
        : <f32> to tensor<32x32x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @wrong_type(
      %a: !tt.ptr<f16>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: element type must be f32}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f16> to tensor<1024x!tt.ptr<f16>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @wrong_extent(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: size must be static extent 1024}}
    %unused = tts.make_tptr %a to sizes: [512], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<512x!tt.ptr<f32>>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @wraparound(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{unsupported structured pointer layout: shape must disable wraparound and broadcasting}}
    %unused = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [1024], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    return
  }
}

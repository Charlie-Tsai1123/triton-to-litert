// RUN: triton-to-litert-opt --normalize-triton-to-litert-launch-metadata --verify-each --verify-diagnostics --split-input-file %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  // expected-error @+1 {{milestone 1 requires a static launch grid of (1, 1, 1)}}
  func.func @missing_launch(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 2, 1, 1>
} {
  // expected-error @+1 {{milestone 1 requires a static launch grid of (1, 1, 1)}}
  func.func @non_unit_launch_x(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 2, 1>
} {
  // expected-error @+1 {{milestone 1 requires a static launch grid of (1, 1, 1)}}
  func.func @non_unit_launch_y(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 2>
} {
  // expected-error @+1 {{milestone 1 requires a static launch grid of (1, 1, 1)}}
  func.func @non_unit_launch_z(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = "dynamic"
} {
  // expected-error @+1 {{milestone 1 requires a static launch grid of (1, 1, 1)}}
  func.func @dynamic_launch(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @unfoldable_launch_derived_expression(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    // expected-error @+1 {{milestone 1 requires the canonical launch offset to have one i32-to-index cast}}
    %unfoldable = arith.index_cast %num_x : i32 to index
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %sum = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%a_value, %b_value : tensor<1024xf32>, tensor<1024xf32>)
        outs(%a_value : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %init: f32):
        %added = arith.addf %lhs, %rhs : f32
        linalg.yield %added : f32
    } -> tensor<1024xf32>
    %out_ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%out_ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

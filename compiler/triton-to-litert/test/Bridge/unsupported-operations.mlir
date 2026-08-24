// RUN: triton-to-litert-opt --verify-triton-to-litert-bridge-input --verify-each --verify-diagnostics --split-input-file %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_triton(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.get_program_id'}}
    %unused = tt.get_program_id x : i32
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @unsupported_tts(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    %ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-error @+1 {{milestone 1 does not support TTS operation 'tts.get_structured_state'}}
    %state:3 = "tts.get_structured_state"(%ptr)
        <{resultSegmentSizes = array<i32: 1, 1, 1>}>
        : (tensor<1024x!tt.ptr<f32>>)
        -> (tensor<1024x!tt.ptr<f32>>, index, index)
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @narrowing(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %wide = arith.constant 0 : i64
    // expected-error @+1 {{semantic narrowing is unsupported at Bridge Input ('arith.trunci')}}
    %unused = arith.trunci %wide : i64 to i32
    return
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>
module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @hidden_operation(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %input = arith.constant dense<0.0> : tensor<1024xf32>
    %unused = linalg.generic {
        indexing_maps = [#identity, #identity],
        iterator_types = ["parallel"]}
        ins(%input : tensor<1024xf32>)
        outs(%input : tensor<1024xf32>) {
      ^bb0(%in: f32, %init: f32):
        // expected-error @+1 {{milestone 1 does not support operation 'arith.mulf'}}
        %product = arith.mulf %in, %in : f32
        linalg.yield %product : f32
    } -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @gather(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %offsets = arith.constant dense<0> : tensor<1024xi32>
    // expected-error @+1 {{milestone 1 does not support unstructured memory operation 'tts.gather'}}
    %unused = tts.gather %a[%offsets]
        : (!tt.ptr<f32>, tensor<1024xi32>) -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @scatter(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %offsets = arith.constant dense<0> : tensor<1024xi32>
    %value = arith.constant dense<0.0> : tensor<1024xf32>
    // expected-error @+1 {{milestone 1 does not support unstructured memory operation 'tts.scatter'}}
    tts.scatter %value into %out[%offsets]
        : tensor<1024xf32> into (!tt.ptr<f32>, tensor<1024xi32>)
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @atomic(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %offsets = arith.constant dense<0> : tensor<1024xi32>
    %value = arith.constant dense<0.0> : tensor<1024xf32>
    // expected-error @+1 {{milestone 1 does not support unstructured memory operation 'tts.unstructured_atomic_rmw'}}
    %unused = tts.unstructured_atomic_rmw fadd, acq_rel, gpu,
        %out[%offsets], %value
        : (!tt.ptr<f32>, tensor<1024xi32>, tensor<1024xf32>)
        -> tensor<1024xf32>
    return
  }
}

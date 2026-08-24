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
  func.func @residual_triton_atomic_rmw(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %value = arith.constant 1.0 : f32
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.atomic_rmw'}}
    %unused = tt.atomic_rmw fadd, acq_rel, gpu, %out, %value
        : (!tt.ptr<f32>, f32) -> f32
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_triton_atomic_cas(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %compare = arith.constant 0.0 : f32
    %value = arith.constant 1.0 : f32
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.atomic_cas'}}
    %unused = tt.atomic_cas acq_rel, gpu, %out, %compare, %value
        : (!tt.ptr<f32>, f32, f32) -> f32
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_triton_atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32) {
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.atomic_poll'}}
    %unused = tt.atomic_poll acquire, gpu, %ptr, %expected
        : !tt.ptr<i32>, i32 -> i1
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_descriptor_load(%desc: !tt.tensordesc<1024xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.descriptor_load'}}
    %unused = tt.descriptor_load %desc[%zero]
        : !tt.tensordesc<1024xf32> -> tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_descriptor_store(
      %desc: !tt.tensordesc<1024xf32>, %value: tensor<1024xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.descriptor_store'}}
    tt.descriptor_store %desc[%zero], %value
        : !tt.tensordesc<1024xf32>, tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_descriptor_reduce(
      %desc: !tt.tensordesc<1024xf32>, %value: tensor<1024xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.descriptor_reduce'}}
    tt.descriptor_reduce add, %desc[%zero], %value
        : !tt.tensordesc<1024xf32>, tensor<1024xf32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_descriptor_gather(
      %desc: !tt.tensordesc<1x128xbf16>, %indices: tensor<32xi32>,
      %offset: i32) {
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.descriptor_gather'}}
    %unused = tt.descriptor_gather %desc[%indices, %offset]
        : (!tt.tensordesc<1x128xbf16>, tensor<32xi32>, i32)
        -> tensor<32x128xbf16>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_descriptor_scatter(
      %desc: !tt.tensordesc<1x128xbf16>, %indices: tensor<32xi32>,
      %offset: i32, %value: tensor<32x128xbf16>) {
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.descriptor_scatter'}}
    tt.descriptor_scatter %desc[%indices, %offset], %value
        : !tt.tensordesc<1x128xbf16>, tensor<32xi32>, i32,
          tensor<32x128xbf16>
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
    %wide = arith.constant 4294967296 : i64
    // expected-error @+1 {{semantic narrowing is unsupported at Bridge Input ('arith.trunci')}}
    %narrow = arith.trunci %wide : i64 to i32
    %offset = arith.index_cast %narrow : i32 to index
    %unused = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%offset], shape: [0], order: []
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
  func.func @unproved_unsigned_index_range(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %wide = arith.constant 4294967296 : i64
    // expected-error @+1 {{semantic narrowing is unsupported at Bridge Input ('arith.index_castui'): source range cannot be proven representable as index}}
    %offset = arith.index_castui %wide : i64 to index
    %unused = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%offset], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
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
  func.func @residual_triton_load(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.load'}}
    %unused = tt.load %a : !tt.ptr<f32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @residual_triton_store(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %value = arith.constant 0.0 : f32
    // expected-error @+1 {{bridge input contains residual Triton operation 'tt.store'}}
    tt.store %out, %value : !tt.ptr<f32>
    return
  }
}

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @unstructured_pointer_recovery(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %offsets = arith.constant dense<0> : tensor<1024xi32>
    // expected-error @+1 {{milestone 1 does not support unstructured memory operation 'tts.make_gather_scatter_tptr'}}
    %unused = tts.make_gather_scatter_tptr %a to sizes: [1024]
        gather_scatter_dim: 0 gather_scatter_offset: %offsets,
        strides: [1], offsets: [0]
        : tensor<1024xi32> <f32> to tensor<1024x!tt.ptr<f32>>
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

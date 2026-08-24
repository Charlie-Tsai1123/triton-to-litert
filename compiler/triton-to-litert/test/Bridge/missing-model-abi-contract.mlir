// RUN: not triton-to-litert-opt --verify-triton-to-litert-bridge-input --verify-each --split-input-file %s 2>&1 | FileCheck %s

module attributes {
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @missing_model_abi_contract(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// CHECK: model ABI cannot be functionalized: expected distinct buffer roles [input, input, output]

// -----

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @missing_noalias_contract(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// CHECK: model ABI cannot be functionalized: expected distinct buffer roles [input, input, output]

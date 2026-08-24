// RUN: not triton-to-litert-opt --verify-triton-to-litert-bridge-input --verify-each %s 2>&1 | FileCheck %s

module {
  func.func @missing_launch_contract(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    return
  }
}

// CHECK: milestone 1 requires a static launch grid of (1, 1, 1)

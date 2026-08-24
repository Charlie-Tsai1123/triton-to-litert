// RUN: triton-to-litert-opt --classify-triton-to-litert-vector-add-linalg --verify-each --verify-diagnostics --split-input-file %s

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @mul_not_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: body arithmetic must be exactly arith.addf}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %product = arith.mulf %lhs, %rhs : f32
        linalg.yield %product : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @reduction_iterator(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: expected exactly one parallel iterator and no reduction iterators}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["reduction"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

module {
  func.func @permuted_maps(
      %a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %empty = tensor.empty() : tensor<32x32xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: expected three exact rank-1 identity indexing maps; broadcast and permutation maps are unsupported}}
    %result = linalg.generic {
        indexing_maps = [#transpose, #transpose, #transpose],
        iterator_types = ["parallel", "parallel"]
      } ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%empty : tensor<32x32xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}

// -----

#scalar = affine_map<(d0) -> ()>
#identity = affine_map<(d0) -> (d0)>

module {
  func.func @broadcast_map(
      %a: tensor<f32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: expected three exact rank-1 identity indexing maps; broadcast and permutation maps are unsupported}}
    %result = linalg.generic {
        indexing_maps = [#scalar, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<f32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func private @side_effect()

  func.func @unsupported_region_operation(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: body must contain only arith.addf followed by linalg.yield}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        func.call @side_effect() : () -> ()
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @wrong_shape(%a: tensor<16xf32>, %b: tensor<16xf32>)
      -> tensor<16xf32> {
    %empty = tensor.empty() : tensor<16xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: all operands and results must be exactly tensor<1024xf32> without an encoding}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<16xf32>, tensor<16xf32>)
        outs(%empty : tensor<16xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<16xf32>
    return %result : tensor<16xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @wrong_type(%a: tensor<1024xi32>, %b: tensor<1024xi32>)
      -> tensor<1024xi32> {
    %empty = tensor.empty() : tensor<1024xi32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: all operands and results must be exactly tensor<1024xf32> without an encoding}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xi32>, tensor<1024xi32>)
        outs(%empty : tensor<1024xi32>) {
      ^bb0(%lhs: i32, %rhs: i32, %unused: i32):
        %sum = arith.addi %lhs, %rhs : i32
        linalg.yield %sum : i32
      } -> tensor<1024xi32>
    return %result : tensor<1024xi32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @swapped_inputs(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: inputs must be the first two model arguments in source order, without swapping or repetition}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%b, %a : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @uses_init_argument(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: the init/output block argument must be unused so the full output domain is proven overwritten}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %init: f32):
        %sum = arith.addf %lhs, %init : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @fast_math(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: arith.addf must have default floating-point semantics}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs fastmath<fast> : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @repeated_inputs(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: inputs must be the first two model arguments in source order, without swapping or repetition}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %a : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @extra_body_operation(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: body must contain only arith.addf followed by linalg.yield}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %ignored = arith.negf %lhs : f32
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @wrong_arity(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %c: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: expected exactly two tensor inputs, one tensor init, and one tensor result}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b, %c : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %third: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @reversed_scalar_inputs(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: arith.addf must consume the two input block arguments in source order}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %rhs, %lhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @wrong_yield(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: linalg.yield must yield only the exact arith.addf result}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %lhs : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @explicit_rounding(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    // expected-error @+1 {{Triton-to-LiteRT cannot classify linalg.generic as milestone-1 elementwise add: arith.addf must have default floating-point semantics}}
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = "arith.addf"(%lhs, %rhs) <{
            fastmath = #arith.fastmath<none>, roundingmode = 1 : i32
          }> : (f32, f32) -> f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

// -----

#identity = affine_map<(d0) -> (d0)>

module {
  func.func @unsupported_surrounding_operation(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>) -> tensor<1024xf32> {
    // expected-error @+1 {{Triton-to-LiteRT functionalized bridge preparation does not support operation 'arith.addf'}}
    %unused_add = arith.addf %a, %b : tensor<1024xf32>
    %empty = tensor.empty() : tensor<1024xf32>
    %result = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused: f32):
        %sum = arith.addf %lhs, %rhs : f32
        linalg.yield %sum : f32
      } -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
}

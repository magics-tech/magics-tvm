/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/qnn/op/add.cc
 * \brief QNN add operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>

#include <algorithm>
#include <cmath>

#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

/*
 * \brief Canonicalizes the QNN add op.
 * \param attrs The empty attribute.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for add op.
 */
Expr QnnAddCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                        const Array<tvm::relay::Type>& arg_types) {
  // Get the args.
  QnnBinaryOpArguments args(new_args);

  // Get the input dtype and shape.
  QnnBinaryOpTensorType input_type(arg_types, 0);

  const auto* broadcast_attrs = attrs.as<BroadcastAttrs>();
  ICHECK(broadcast_attrs != nullptr);

  auto lhs_axis = broadcast_attrs->lhs_axis;
  auto rhs_axis = broadcast_attrs->rhs_axis;

  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);

  // When each input scale is an integer multiple of the smaller one, min_scale, the inputs add
  // exactly at min_scale, so the sum is requantized once and rounds once. With
  // scale_a = r_a * min_scale and scale_b = r_b * min_scale,
  //
  //          scale_c * (Q_c - zp_c) = min_scale * (r_a * (Q_a - zp_a) + r_b * (Q_b - zp_b))
  //
  // so Q_c is r_a * (Q_a - zp_a) + r_b * (Q_b - zp_b), requantized from (min_scale, 0) to
  // (scale_c, zp_c).
  if (IsConstScalar(args.lhs_scale) && IsConstScalar(args.rhs_scale) &&
      IsConstScalar(args.lhs_zero_point) && IsConstScalar(args.rhs_zero_point)) {
    float lhs_scale = GetScalarFromConstant<float>(args.lhs_scale);
    float rhs_scale = GetScalarFromConstant<float>(args.rhs_scale);
    float min_scale = std::min(lhs_scale, rhs_scale);
    double lhs_ratio = lhs_scale / min_scale;
    double rhs_ratio = rhs_scale / min_scale;
    // Up to 2^14, so r * (Q - zp) fits int32 even for int16 inputs
    if (lhs_ratio == std::round(lhs_ratio) && rhs_ratio == std::round(rhs_ratio) &&
        std::max(lhs_ratio, rhs_ratio) <= 16384) {
      auto lhs = Multiply(Subtract(Cast(args.lhs, DataType::Int(32)), args.lhs_zero_point),
                          MakeConstantScalar(DataType::Int(32), static_cast<int32_t>(lhs_ratio)));
      auto rhs = Multiply(Subtract(Cast(args.rhs, DataType::Int(32)), args.rhs_zero_point),
                          MakeConstantScalar(DataType::Int(32), static_cast<int32_t>(rhs_ratio)));
      auto output =
          RequantizeOrUpcast(Add(lhs, rhs), MakeConstantScalar(DataType::Float(32), min_scale),
                             zero_scalar, args.output_scale, args.output_zero_point,
                             input_type.shape);
      return ConvertDtype(output, input_type.dtype);
    }
  }

  // Since the input qnn params can be different than output qnn params, we first requantize the
  // input tensors to the output qnn params. Then we call relay.add on the requantized inputs. This
  // addition results in extra addition of the output zero point. We further subtract the zero
  // point. The whole process can be represented using following equations
  //
  //          scale_c * (Q_c - zp_c) = scale_a * (Q_a - zp_a) + scale_b * (Q_b - zp_b)
  //
  // After requantizing Q_a and Q_b, equation becomes,
  //          scale_c * (Q_c - zp_c) = scale_c * (Q_a' - zp_c) + scale_c * (Q_b' - zp_c)
  //          scale_c * (Q_c - zp_c) = scale_c * (Q_a' + Q_b' - zp_c - zp_c)
  //
  // Comparing the LHS and RHS, it results in
  //          Q_c = Q_a' + Q_b' - zp_c
  // The add op is done in int32 precision.

  // Requantize LHS if necessary. Computes Q_a'
  auto requantized_lhs =
      RequantizeOrUpcast(args.lhs, args.lhs_scale, args.lhs_zero_point, args.output_scale,
                         args.output_zero_point, input_type.shape, lhs_axis);
  // Requantize RHS if necessary. Computes Q_b'
  auto requantized_rhs =
      RequantizeOrUpcast(args.rhs, args.rhs_scale, args.rhs_zero_point, args.output_scale,
                         args.output_zero_point, input_type.shape, rhs_axis);
  // Computes Q_a' + Q_b'
  auto output = Add(requantized_lhs, requantized_rhs);

  // Subtract zero point. Computes (Q_a' + Q_b') - zp_c
  if (!IsEqualScalar(args.output_zero_point, zero_scalar)) {
    output = Subtract(output, args.output_zero_point);
  }

  // Go back to lower precision.
  return ConvertDtype(output, input_type.dtype);
}

// QNN Addition operator.
QNN_REGISTER_BINARY_OP("add")
    .describe("Elementwise add with broadcasting for quantized tensors.")
    .set_support_level(11)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnAddCanonicalize)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm

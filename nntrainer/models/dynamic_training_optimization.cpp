// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   dynamic_training_optimization.cpp
 * @date   5 January 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Dynamic Training Optimization for Neural Network
 *
 */

#include <numeric>
#include <random>
#include <vector>

#include <dynamic_training_optimization.h>
#include <optimizer_wrapped.h>
#include <tensor.h>
#include <util_func.h>
#include <weight.h>

namespace nntrainer {
DynamicTrainingOptimization::DynamicTrainingOptimization(float threshold_,
                                                         int skip_n_iter) :
  threshold(threshold_),
  enabled(false),
  epsilon(1e-7f),
  skip_n_iterations(skip_n_iter) {
  reduce_op = reduceByNorm;
  calc_ratio_op = ratioUsingDerivative;
  // rng.seed(getSeed());
  rng.seed(0);
  dist = std::uniform_real_distribution<float>(0.0, 1.0);
}

/**
 * @brief     Check if the given weights can skip updating
 * @note true if should be applied, else false
 */
bool DynamicTrainingOptimization::checkIfApply(
  const std::vector<Weight> &weights, const std::shared_ptr<Var_Grad> &input,
  const std::shared_ptr<Var_Grad> &output,
  const std::shared_ptr<OptimizerWrapped> &opt, int iteration) {
  if (!enabled || iteration < skip_n_iterations)
    return true;

  std::vector<bool> apply;
  apply.reserve(weights.size());

  for (auto const &weight : weights)
    apply.push_back(checkIfApply(weight, input, output, opt, iteration));

  return std::accumulate(apply.begin(), apply.end(), true,
                         std::logical_and<bool>());
}

/**
 * @brief     Check if the given weight can skip updating
 * @note true if should be applied, else false
 */
bool DynamicTrainingOptimization::checkIfApply(
  const Weight &weight, const std::shared_ptr<Var_Grad> &input,
  const std::shared_ptr<Var_Grad> &output,
  const std::shared_ptr<OptimizerWrapped> &opt, int iteration) {
  if (iteration < skip_n_iterations)
    return true;

  if (!weight.hasGradient() || weight.getGradientRef().empty())
    return true;

  float reduced_ratio = calc_ratio_op(weight, input, output, reduce_op);

  return checkIfApply(reduced_ratio, (float)opt->getLearningRate(iteration));
}

/**
 * @brief   Calculate the ratio of update to the weight using derivative
 */
float DynamicTrainingOptimization::ratioUsingDerivative(
  const Weight &weight, const std::shared_ptr<Var_Grad> &input,
  const std::shared_ptr<Var_Grad> &output,
  std::function<float(Tensor const &)> reduce_op) {
  float reduced_derivative = reduce_op(output->getGradientRef());
  float reduced_input = reduce_op(input->getVariableRef());
  float reduced_weight = reduce_op(weight.getVariableRef());
  float reduced_grad = reduced_derivative * reduced_input;

  return reduced_grad / reduced_weight;
}

/**
 * @brief   Calculate the ratio of update to the weight using gradient
 */
float DynamicTrainingOptimization::ratioUsingGradient(
  const Weight &weight, const std::shared_ptr<Var_Grad> &input,
  const std::shared_ptr<Var_Grad> &output,
  std::function<float(Tensor const &)> reduce_op) {
  Tensor ratio = weight.getGradientRef().divide(weight.getVariableRef());
  return reduce_op(ratio);
}

/**
 * @brief   Check if the update should be applied or skipped
 * @note true if should be applied, else false
 */
bool DynamicTrainingOptimization::checkIfApply(float reduced_ratio,
                                               float learning_rate) {
  /**
   * If the reduced update ratio is higher than 1, then always apply update.
   * If the reduced update ratio is less than 1, then apply it with
   * probability = update ratio
   */
  if (dist(rng) < reduced_ratio * learning_rate / threshold)
    return true;

  return false;
}

/**
 * @brief     Operation to decide if update should be skipped
 * @note      Calculate l0 norm of the tensor
 */
float DynamicTrainingOptimization::reduceByMax(Tensor const &ratio) {
  return ratio.max_abs();
}

/**
 * @brief     Operation to decide if update should be skipped
 * @note      Calculate l2 norm of the tensor averaged by its size
 */
float DynamicTrainingOptimization::reduceByNorm(Tensor const &ratio) {
  float l2norm = ratio.l2norm();
  return l2norm / std::sqrt(ratio.size());
}

} /* namespace nntrainer */

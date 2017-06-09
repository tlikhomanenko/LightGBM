#ifndef LIGHTGBM_BOOSTING_INFINITEBOOST_H_
#define LIGHTGBM_BOOSTING_INFINITEBOOST_H_

#include <LightGBM/boosting.h>
#include "score_updater.hpp"
#include "gbdt.h"

#include <cstdio>
#include <vector>
#include <string>
#include <fstream>

namespace LightGBM {
/*!
* \brief InfiniteBoost algorithm https://arxiv.org/pdf/1706.01109.pdf implementation including training, prediction.
*/
class InfiniteBoost: public GBDT {
public:
  /*!
  * \brief Constructor
  */
  InfiniteBoost() : GBDT() { }
  /*!
  * \brief Destructor
  */
  ~InfiniteBoost() { }
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
            const std::vector<const Metric*>& training_metrics) override {
    GBDT::Init(config, train_data, objective_function, training_metrics);
    // Log::Info("lr=%f", shrinkage_rate_);
    capacity_ = gbdt_config_->capacity;
    // ignore shrinkage during ensemble construction
    shrinkage_rate_ = 1.f;
    normalization_ = 0.0f;
    // TODO this is an analytic expression
    for (int i = 0; i < gbdt_config_->num_iterations; ++i) {
      normalization_ += (i + 1);
    }
    current_normalization_ = 0.;
  }

  // TODO there is no point in this function
  void ResetTrainingData(const BoostingConfig* config, const Dataset* train_data, const ObjectiveFunction* objective_function,
                         const std::vector<const Metric*>& training_metrics) override {
    GBDT::ResetTrainingData(config, train_data, objective_function, training_metrics);
  }
  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t* gradient, const score_t* hessian, bool is_eval) override {
    GBDT::TrainOneIter(gradient, hessian, false);
    // normalize trees in the ensemble
    UpdateTreeWeight();
    if (is_eval) {
      auto best_msg = OutputMetric(iter_);
    }
    return false;
  }
  
private:
  /*!
  * \brief put necessary coefficient for trained tree_m: capacity * m / sum (1 + .. + n_iterations)
  */
  void UpdateTreeWeight() {
    double eta = 2. / (iter_ + 1);
    double tree_contribution = std::min(eta * capacity_, 1.);
    // update current normalization
    current_normalization_ += iter_;

    // update scores using formula F \to (1 - eta_m) * F + eta_m * capacity * tree_m
    // to avoid large contribution of the new tree at the initial boosting iterations take min(eta_m * capacity, 1)
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto last_tree = (iter_ - 1) * num_tree_per_iteration_ + cur_tree_id;
      // Removing the contribution added by GBDT
      models_[last_tree]->Shrinkage(-1);
      // update score on the validation set
      for (auto& score_updater : valid_score_updater_) {
        // remove the latest tree predictions from the score
        score_updater->AddScore(models_[last_tree].get(), cur_tree_id);
        // update current score
        score_updater->MultiplyScore(1 - eta, cur_tree_id);
      }
      // update score on the training set
      // remove the latest tree predictions from the score
      train_score_updater_->AddScore(models_[last_tree].get(), cur_tree_id);
      // update current score
      train_score_updater_->MultiplyScore(1 - eta, cur_tree_id);
    }
    
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      auto last_tree = (iter_ - 1) * num_tree_per_iteration_ + cur_tree_id;
      // take minus coefficient to compensate -1 multiplication at the beginning
      models_[last_tree]->Shrinkage(-tree_contribution);
      for (auto& score_updater : valid_score_updater_) {
        score_updater->AddScore(models_[last_tree].get(), cur_tree_id);
      }
      train_score_updater_->AddScore(models_[last_tree].get(), cur_tree_id);
      
      // set the final contribution of the tree_m: capacity * m / sum (1 + .. + n_iterations)
      models_[last_tree]->Shrinkage(1. / tree_contribution * std::min(
        capacity_ * iter_ / normalization_, current_normalization_ / normalization_)
      );
    }
  }
  double capacity_;
  // 1 + 2 + 3 + .. + n_iterations
  double normalization_;
  // 1 + 2 + 3 + .. + current_iteration
  double current_normalization_;
};

}  // namespace LightGBM
#endif   // LIGHTGBM_BOOSTING_INFINITEBOOST_H_

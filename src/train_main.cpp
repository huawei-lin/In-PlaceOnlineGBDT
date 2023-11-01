// Copyright 2022 The ONLINEGBDT Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "data.h"
#include "model.h"
#include "model_gpu.h"
#include "tree.h"
#include "utils.h"

int main(int argc, char* argv[]) {
  std::unique_ptr<ONLINEGBDT::Config> config =
      std::unique_ptr<ONLINEGBDT::Config>(new ONLINEGBDT::Config());
  config->parseArguments(argc, argv);
  config->model_mode = "train";

  ONLINEGBDT::ModelHeader model_header =
      ONLINEGBDT::GradientBoosting::loadModelHeader(config.get());
  int start_epoch = 0;
  if (model_header.config.null_config == false) {
    *config = model_header.config;
    config->parseArguments(argc, argv);
    config->model_mode = "train";
    start_epoch = model_header.config.model_n_iterations;
  }
  if (config->tree_max_n_leaves <= ONLINEGBDT::Utils::ipow(2, config->tree_n_random_layers)) {
    fprintf(stderr, "Error: tree_max_n_leaves must be greater than 2^${tree_n_random_layers}\n");
    exit(1);
  }

  config->sanityCheck();

  std::unique_ptr<ONLINEGBDT::Data> data =
      std::unique_ptr<ONLINEGBDT::Data>(new ONLINEGBDT::Data(config.get()));
  if (model_header.config.null_config == false){
    data->data_header = model_header.auxDataHeader;
    data->loadData(false);
  }else{
    data->loadData(true);
  }
  data->constructAuxData();

  std::unique_ptr<ONLINEGBDT::GradientBoosting> model;

  if (config->model_name != "regression"){
    config->model_use_logit = (config->model_name.find("logit") != std::string::npos);
  }

  if (config->model_name == "mart" || config->model_name == "robustlogit") {
#ifdef CUDA
    if (config->use_gpu){
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::MartGPU(data.get(), config.get()));
    }else{
//      if(data->data_header.n_classes == 2){
//        model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
//            new ONLINEGBDT::BinaryMart(data.get(), config.get()));
//      }else{
//        model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
//            new ONLINEGBDT::Mart(data.get(), config.get()));
//      }
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::Mart(data.get(), config.get()));
    }
#else
 //   if(data->data_header.n_classes == 2){
 //     model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
 //         new ONLINEGBDT::BinaryMart(data.get(), config.get()));
 //   }else{
 //     model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
 //         new ONLINEGBDT::Mart(data.get(), config.get()));
 //   }
    model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
        new ONLINEGBDT::Mart(data.get(), config.get()));
#endif
  } else if (config->model_name == "abcmart" ||
             config->model_name == "abcrobustlogit") {
#ifdef CUDA
    if(config->use_gpu){
	  if(data->data_header.n_classes == 2){
        model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
            new ONLINEGBDT::MartGPU(data.get(), config.get()));		  
	  }else{
        model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
            new ONLINEGBDT::ABCMartGPU(data.get(), config.get()));
	  }
    }else{
      if(data->data_header.n_classes == 2){
        model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
            new ONLINEGBDT::BinaryMart(data.get(), config.get()));
      }else{
        model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
            new ONLINEGBDT::ABCMart(data.get(), config.get()));
      }
    }
#else
    if(data->data_header.n_classes == 2){
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::BinaryMart(data.get(), config.get()));
    }else{
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::ABCMart(data.get(), config.get()));
    }
#endif
  } else if (config->model_name == "regression") {
    config->model_is_regression = true;
#ifdef CUDA
    if(config->use_gpu){
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::RegressionGPU(data.get(), config.get()));
    }else{
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::Regression(data.get(), config.get()));
    }
#else
    model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
        new ONLINEGBDT::Regression(data.get(), config.get()));
#endif
  } else if (config->model_name == "lambdamart" || config->model_name == "lambdarank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
				new ONLINEGBDT::LambdaMart(data.get(), config.get()));
  } else if (config->model_name == "gbrank") {
    config->model_is_regression = 1;
		config->model_use_logit = true;
    model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
				new ONLINEGBDT::GBRank(data.get(), config.get()));
  } else {
    fprintf(stderr, "Unsupported model name %s\n", config->model_name.c_str());
    exit(1);
  }
  model->init();
  model->loadModel();
  model->setupExperiment();
  if (start_epoch > 0) {
    model->start_epoch = start_epoch;
    int old_n_iter = model->getConfig()->model_n_iterations;
    model->getConfig()->model_n_iterations = start_epoch;
    fprintf(stderr,"[Info] Reconstruct F from the save model.\n");
    model->test();
    fprintf(stderr,"[Info] F reconstruction finished.\n");
    model->getConfig()->model_n_iterations = old_n_iter;
  }

  model->train();
  return 0;
}


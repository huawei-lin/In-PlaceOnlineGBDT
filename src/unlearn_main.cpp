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
  config->model_mode = "unlearn";

  ONLINEGBDT::ModelHeader model_header =
      ONLINEGBDT::GradientBoosting::loadModelHeader(config.get());
  if (model_header.config.null_config == false) {
    *config = model_header.config;
    config->parseArguments(argc, argv);
    config->model_mode = "unlearn";
  } else {
    printf("[ERROR] Model file not found: -model (%s)\n",config->model_pretrained_path.c_str());
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

  std::vector<uint> unids;
  data->loadUnlearningIndies(config->unlearning_ids_path, unids);

  std::unique_ptr<ONLINEGBDT::GradientBoosting> model;

  // config->model_use_logit =
  //     (config->model_name.find("logit") != std::string::npos);

  if (config->model_name == "mart" || config->model_name == "robustlogit") {
    /*
    if(data->data_header.n_classes == 2){
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::BinaryMart(data.get(), config.get()));
    }else{
    */ 
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::Mart(data.get(), config.get()));
  /*
    }
  } else if (config->model_name == "abcmart" ||
             config->model_name == "abcrobustlogit") {
    if(data->data_header.n_classes == 2){
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::BinaryMart(data.get(), config.get()));
    }else{
      model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
          new ONLINEGBDT::ABCMart(data.get(), config.get()));
    }
  } else if (config->model_name == "regression") {
    config->model_is_regression = true;
    model = std::unique_ptr<ONLINEGBDT::GradientBoosting>(
        new ONLINEGBDT::Regression(data.get(), config.get()));
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
  */
  } else {
    fprintf(stderr, "Unsupported model name %s\n", config->model_name.c_str());
    exit(1);
  }

  model->init();
  model->loadModel();
  model->setupExperiment();

  model->unlearn(unids);
  return 0;
}

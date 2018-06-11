# There is a strong overlap between (train_model(), train_model_multiple())
# and (train_model_hierarchical(), train_model_hierarchical_multiple)

generate_target = function(target, f_id, WINDOW_SIZE, SAMPLING_SIZE = 10){
  # print(f_id)
  target_tmp = target[ForecastId == f_id]
  
  WINDOW_SIZE = seq_len(WINDOW_SIZE)
  
  target_tmp = target_tmp[, shift(.SD, WINDOW_SIZE, NA, "lead", TRUE), .SDcols=c("Timestamp", "Value")] %>% 
    bind_cols(target_tmp %>% 
                select(-Value), .) 
  
  colA = paste("Timestamp_lead_", WINDOW_SIZE, sep = "")
  colB = paste("Value_lead_", WINDOW_SIZE, sep = "")
  target_tmp = melt.data.table(target_tmp, measure.vars = list(colA, colB), 
                               value.name = c("Timestamp_lead", "Value_lead"), variable.name = "lead",
                               variable.factor = F)
  
  target_tmp = target_tmp[ , lead := as.integer(lead)]
  
  # Discard asap useless lines
  target_tmp = target_tmp[!is.na(Value_lead)]
  
  # Try to subsample smartly, given metric
  n_obs = 200
  target_tmp = target_tmp[ , weight :=  (3 * n_obs - (2 * lead) + 1) / (2 * (n_obs ** 2))]
  
  SAMPLING_SIZE = max(WINDOW_SIZE) * SAMPLING_SIZE
  set.seed(1)
  target_tmp = target_tmp[sample(.N,SAMPLING_SIZE, prob = weight, replace = T)] # Bottleneck when replace = F
  
  return(target_tmp)
}

generate_target_loop = function(target, f_id, WINDOW_SIZE, SAMPLING_SIZE = 10){
  # Avoid memory overload
  res = vector("list", length = length(f_id))
  pb = progress_estimated(length(f_id))
  for(i in seq_along(f_id)){
    pb$pause(0)$tick()$print()
    res[[i]] = generate_target(target, f_id[i], WINDOW_SIZE[i], SAMPLING_SIZE = SAMPLING_SIZE)
  }
  res = bind_rows(res)
}

generate_target_test = function(train, submission_format){
  start = train %>% 
    group_by(SiteId, ForecastId) %>% 
    summarise(Timestamp = last(Timestamp)) %>% 
    ungroup()
  
  end = submission_format %>% 
    group_by(SiteId, ForecastId) %>% 
    mutate(lead = row_number()) %>% 
    ungroup() %>% 
    rename(Timestamp_lead = Timestamp) %>% 
    select(ForecastId, Timestamp_lead, lead)
  
  start %>% 
    inner_join(end, by = "ForecastId")
}

get_value_mean_before = function(v_ForecastId, v_SiteId){
  train %>% 
    filter(ForecastId <= v_ForecastId, SiteId == v_SiteId) %>% 
    pull(Value) %>% mean(na.rm = T)
}

train_model = function(df_learning, train_full = T, verbose = -1){
  # df_learning - the dataset to learn from
  # train_full - whether to retrain the model on full data. 
  # Use it as True for submission and False for local tuning
  df_learning_train = df_learning %>% 
    filter(set == "train") %>% 
    filter(!is.na(Value_lead))
  
  NFOLD = 5
  df_learning_train = df_learning_train %>% 
    group_by(ForecastId) %>% 
    arrange(Timestamp_lead) %>% 
    mutate(folders = ntile(row_number(), NFOLD)) %>% 
    ungroup()
  
  df_train = df_learning_train %>% 
    filter(folders != max(folders)) %>% 
    select(-folders)
  
  # df_train = df_train %>%
  #   sample_frac(0.5)
  
  df_valid = df_learning_train %>% 
    filter(folders == max(folders)) %>% 
    select(-folders)
  
  rm(df_learning_train)
  
  df_test = df_learning %>%
    filter(set == "test")
  
  # print("Modeling")
  to_drop = c("SiteId", "ForecastId", "Timestamp_weather", "Timestamp", "Timestamp_lead", 
              "set", "weight", "Timestamp_last_day", "Timestamp_last_dow", "size", 
              "timestep_minutes", "Timestamp_last_closed")
  
  to_drop = c(to_drop , "Surface", "Sampling", "BaseTemperature") # Constant for same SiteId
  
  categorical_feature = c()
  
  target = "Value_lead"
  
  X_train = df_train %>%
    select(-one_of(c(to_drop, target))) %>%
    as.matrix()
  
  X_valid = df_valid %>%
    select(-one_of(c(to_drop, target))) %>%
    as.matrix()
  
  X_test = df_test %>%
    select(-one_of(c(to_drop, target))) %>%
    as.matrix()
  
  y_train = df_train %>% pull(target)
  y_valid = df_valid %>% pull(target)
  
  # weight_train = df_train %>% pull(weight) 
  # weight_valid = df_valid %>% pull(weight)
  
  params <- list(num_leaves = 2^5-2
                 ,objective = "regression_l2"
                 ,min_data_in_leaf = 20
                 ,learning_rate = 0.1
                 ,metric = "l2_root")
  
  lgb_train = lgb.Dataset(X_train, label = y_train) #, weight = weight_train)
  lgb_valid = lgb.Dataset(X_valid, label = y_valid) #, weight = weight_valid)
  
  model_lgb <- lgb.train(params, lgb_train, nrounds = 50000,
                         categorical_feature = categorical_feature,
                         valids = list(train = lgb_train, valid=lgb_valid),
                         early_stopping_rounds = 100, verbose = verbose, eval_freq = 50)
  
  if(train_full == T){
    lgb_full = lgb.Dataset(rbind(X_train, X_valid), label = c(y_train, y_valid)) #, weight = weight_valid)
    model_lgb_full <- lgb.train(params, lgb_full, nrounds = model_lgb$best_iter,
                                categorical_feature = categorical_feature, verbose = -1)
  }else{
    model_lgb_full = model_lgb
  }
  
  if(verbose == 1){
    print("Metrics performance")
    model_lgb$best_score %>% abs() %>% print()
    (df_valid$weight %>% mean() * model_lgb$best_score %>% abs()) %>% print()
    
    print(paste0("Best iter : ", model_lgb$best_iter))
    
    importance_matrix = lgb.importance(model = model_lgb)
    importance_matrix %>%
      filter(Gain > 0.001) %>%
      mutate(Gain = round(Gain, 3)) %>% print()
  }
  
  post_process <- . %>%
    mutate(Value_lead_pred = pmin(Value_lead_pred, max(c(y_train, y_valid)))) %>% # Clip by max(Value_lead)
    mutate(Value_lead_pred = pmax(Value_lead_pred, min(c(y_train, y_valid)))) %>% # Clip min(Value_lead)
    mutate(Value = Value_lead_pred * value_last_mean)
  
  submission = df_test %>%
    select(SiteId, ForecastId, Timestamp_lead, value_last_mean, lead) %>% 
    mutate(Value_lead_pred = predict(model_lgb_full, X_test))
  
  df_valid_pred = df_valid %>% 
    select(SiteId, ForecastId, Timestamp_lead, Value_lead, value_last_mean, weight, lead, size) %>% 
    mutate(Value_lead_pred = predict(model_lgb, X_valid))
  
  df_train_pred = df_train %>% 
    select(SiteId, ForecastId, Timestamp_lead, Value_lead, value_last_mean, weight, lead, size) %>% 
    mutate(Value_lead_pred = predict(model_lgb, X_train))
  
  submission = submission %>% 
    post_process()
  
  df_train_pred = df_train_pred %>% 
    post_process()
  
  df_valid_pred = df_valid_pred %>% 
    post_process()
  
  return(list(submission = submission, df_valid_pred = df_valid_pred, 
              df_train_pred = df_train_pred))
}

train_model_hierarchical = function(df_learning, train_full = T, verbose = -1){
  iter_siteId_tmp = df_learning$SiteId %>% unique %>% sort()
  iter_siteId_tmp = iter_siteId_tmp
  res_predict = vector("list", length = length(iter_siteId_tmp))
  pb = progress_estimated(length(iter_siteId_tmp))
  for(i in seq_len(length(iter_siteId_tmp))){
    pb$pause(0)$tick()$print()
    df_learning_tmp = df_learning %>% filter(SiteId == iter_siteId_tmp[i])
    res_predict[[i]] = train_model(df_learning_tmp, train_full = train_full, verbose = verbose)
  }
  submission = lapply(res_predict, `[[`, 1) %>% 
    rbindlist()
  
  df_valid_pred = lapply(res_predict, `[[`, 2) %>% 
    rbindlist()
  
  df_train_pred = lapply(res_predict, `[[`, 3) %>% 
    rbindlist()
  
  df_valid_pred %>% 
    compute_loss() %>% print()
  
  return(list(submission = submission, df_valid_pred = df_valid_pred, 
              df_train_pred = df_train_pred))
}

train_model_multiple = function(df_learning, train_full = T){
  # df_learning - the dataset to learn from
  # train_full - whether to retrain the model on full data. 
  # Use it as True for submission and False for local tuning
  df_learning_train = df_learning %>% 
    filter(set == "train") %>% 
    filter(!is.na(Value_lead))
  
  NFOLD = 5
  df_learning_train = df_learning_train %>% 
    group_by(ForecastId) %>% 
    arrange(Timestamp_lead) %>% 
    mutate(folders = ntile(row_number(), NFOLD)) %>% 
    ungroup()
  
  df_train = df_learning_train %>% 
    filter(folders != max(folders)) %>% 
    select(-folders)
  
  df_valid = df_learning_train %>% 
    filter(folders == max(folders)) %>% 
    select(-folders)
  
  rm(df_learning_train)
  
  df_test = df_learning %>%
    filter(set == "test")
  
  # print("Modeling")
  to_drop = c("SiteId", "ForecastId", "Timestamp_weather", "Timestamp", "Timestamp_lead", 
              "set", "weight", "Timestamp_last_day", "Timestamp_last_dow", "size", 
              "timestep_minutes", "Timestamp_last_closed")
  
  to_drop = c(to_drop , "Surface", "Sampling", "BaseTemperature") # Constant for same SiteId
  
  categorical_feature = c()
  
  target = "Value_lead"
  
  X_train = df_train %>%
    select(-one_of(c(to_drop, target))) %>%
    as.matrix()
  
  X_valid = df_valid %>%
    select(-one_of(c(to_drop, target))) %>%
    as.matrix()
  
  X_test = df_test %>%
    select(-one_of(c(to_drop, target))) %>%
    as.matrix()
  
  y_train = df_train %>% pull(target)
  y_valid = df_valid %>% pull(target)
  
  # params <- list(num_leaves = 2^5-2
  #                ,objective = "regression_l2"
  #                # ,max_depth = 8
  #                ,min_data_in_leaf = 5
  #                ,learning_rate = 0.1
  #                ,metric = "l2_root")
  
  params1 = list(num_leaves = 2^3
                 ,objective = "regression_l2"
                 # ,max_depth = 4
                 # ,min_data_in_leaf = 200
                 ,learning_rate = 0.1
                 ,metric = "l2_root")
  
  params2 = list(num_leaves = 2^5-2
                 ,objective = "regression_l2"
                 # ,max_depth = 8
                 # ,min_data_in_leaf = 200
                 ,learning_rate = 0.1
                 ,metric = "l2_root")
  
  params3 = list(num_leaves = 2^6
                 ,objective = "regression_l2"
                 # ,max_depth = 12
                 # ,min_data_in_leaf = 200
                 ,learning_rate = 0.1
                 ,metric = "l2_root")
  
  params4 = list(num_leaves = 2^2
                 ,objective = "regression_l2"
                 # ,max_depth = 12
                 # ,min_data_in_leaf = 200
                 ,learning_rate = 0.1
                 ,metric = "l2_root")
  
  list_params = list(params1,
                     params2,
                     params3,
                     params4)
  
  lgb_train = lgb.Dataset(X_train, label = y_train) #, weight = weight_train)
  lgb_valid = lgb.Dataset(X_valid, label = y_valid) #, weight = weight_valid)
  
  submission = df_test %>%
    select(SiteId, ForecastId, Timestamp_lead, value_last_mean, lead) 
  
  df_valid_pred = df_valid %>% 
    select(SiteId, ForecastId, Timestamp_lead, Value_lead, value_last_mean, weight, lead, size)
  
  df_train_pred = df_train %>% 
    select(SiteId, ForecastId, Timestamp_lead, Value_lead, value_last_mean, weight, lead, size) 
  
  for(i in seq_along(list_params)){
    params = list_params[[i]]
    
    model_lgb <- lgb.train(params, lgb_train, nrounds = 20000,
                           categorical_feature = categorical_feature,
                           valids = list(train = lgb_train, valid=lgb_valid),
                           early_stopping_rounds = 100, verbose = -1)
    
    if(train_full == T){
      lgb_full = lgb.Dataset(rbind(X_train, X_valid), label = c(y_train, y_valid)) #, weight = weight_valid)
      model_lgb_full <- lgb.train(params, lgb_full, nrounds = model_lgb$best_iter,
                                  categorical_feature = categorical_feature, verbose = -1)
    }else{
      model_lgb_full = model_lgb
    }
    
    verbose = -1
    if(verbose == 1){
      print("Metrics performance")
      model_lgb$best_score %>% abs() %>% print()
      (df_valid$weight %>% mean() * model_lgb$best_score %>% abs()) %>% print()
      
      print(paste0("Best iter : ", model_lgb$best_iter))
      
      importance_matrix = lgb.importance(model = model_lgb)
      importance_matrix %>%
        filter(Gain > 0.001) %>%
        mutate(Gain = round(Gain, 3))
    }

    submission = submission %>% 
      mutate(!!paste0("Value_lead_pred", "_", i) := predict(model_lgb_full, X_test))
    
    df_valid_pred = df_valid_pred %>% 
      mutate(!!paste0("Value_lead_pred", "_", i) := predict(model_lgb, X_valid))
    
    df_train_pred = df_train_pred %>% 
      mutate(!!paste0("Value_lead_pred", "_", i) := predict(model_lgb, X_train))
  }

  df_valid_pred = df_valid_pred %>% 
    mutate(Value_lead_pred = (Value_lead_pred_1 + Value_lead_pred_2 + Value_lead_pred_3)/3)
  
  df_train_pred = df_train_pred %>% 
    mutate(Value_lead_pred = (Value_lead_pred_1 + Value_lead_pred_2 + Value_lead_pred_3)/3)
  
  submission = submission %>% 
    mutate(Value_lead_pred = (Value_lead_pred_1 + Value_lead_pred_2 + Value_lead_pred_3)/3)
  
  post_process <- .  %>%
    mutate(Value_lead_pred = pmin(Value_lead_pred, max(c(y_train, y_valid)))) %>% # Clip by max(Value_lead)
    mutate(Value_lead_pred = pmax(Value_lead_pred, min(c(y_train, y_valid)))) %>% # Clip min(Value_lead)
    mutate(Value = Value_lead_pred * value_last_mean)
  
  submission = submission %>% 
    post_process()
  
  df_train_pred = df_train_pred %>% 
    post_process()
  
  df_valid_pred = df_valid_pred %>% 
    post_process()
  
  return(list(submission = submission, df_valid_pred = df_valid_pred, 
              df_train_pred = df_train_pred))
}

train_model_hierarchical_multiple = function(df_learning, train_full = T){
  iter_siteId_tmp = df_learning$SiteId %>% unique %>% sort()
  iter_siteId_tmp = iter_siteId_tmp
  res_predict = vector("list", length = length(iter_siteId_tmp))
  pb = progress_estimated(length(iter_siteId_tmp))
  for(i in seq_len(length(iter_siteId_tmp))){
    pb$pause(0)$tick()$print()
    df_learning_tmp = df_learning %>% filter(SiteId == iter_siteId_tmp[i])
    res_predict[[i]] = train_model_multiple(df_learning_tmp, train_full = train_full)
  }
  submission = lapply(res_predict, `[[`, 1) %>% 
    rbindlist()
  
  df_valid_pred = lapply(res_predict, `[[`, 2) %>% 
    rbindlist()
  
  df_train_pred = lapply(res_predict, `[[`, 3) %>% 
    rbindlist()
  
  df_valid_pred %>% 
    compute_loss() %>% print()
  
  return(list(submission = submission, df_valid_pred = df_valid_pred, 
              df_train_pred = df_train_pred))
}

compute_loss = function(df_pred){
  df_pred %>% 
    as_tibble() %>% 
    group_by(SiteId, ForecastId) %>%
    summarise(metrics = Metrics::rmse(actual = Value_lead*weight, predicted = Value_lead_pred*weight),
              size = first(size)) %>% 
    ungroup() %>% 
    summarise(metrics = weighted.mean(metrics, size))
}

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

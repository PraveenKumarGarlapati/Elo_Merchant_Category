library(xgboost)
library(lubridate)
library(magrittr)
library(tidyverse)


#---------------------------
cat("Preprocessing historical transactions...\n")

hist_trnx = hist_trnx%>%
  mutate(purchase_amount = round(purchase_amount/ 0.00150265118 + 497.06,8))
  
htrans = hist_trnx

#htrans <- read_csv("../input/historical_transactions.csv") %>% 
htrans = htrans%>% rename(card = card_id)

sum_htrans_id <- htrans %>%
  group_by(card) %>%
  summarise_at(vars(ends_with("_id")), n_distinct, na.rm = TRUE) 

ohe_htrans <- htrans %>%
  select(authorized_flag, starts_with("category")) %>% 
  mutate_all(factor) %>% 
  model.matrix.lm(~ . - 1, ., na.action = NULL) %>% 
  as_tibble()

fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))

sum_htrans <- htrans %>%
  select(-authorized_flag, -starts_with("category"), -ends_with("_id")) %>% 
  add_count(card) %>%
  group_by(card) %>%
  mutate(date_diff = as.integer(diff(range(purchase_date))),
         prop = n() / sum(n)) %>% 
  ungroup() %>% 
  mutate(year = year(purchase_date),
         month = month(purchase_date),
         day = day(purchase_date),
         hour = hour(purchase_date),
         month_diff = as.integer(ymd("2018-12-01") - date(purchase_date)) / 30 + month_lag) %>% 
  select(-purchase_date) %>% 
  bind_cols(ohe_htrans) %>% 
  group_by(card) %>%
  summarise_all(fn) %>% 
  left_join(sum_htrans_id)

  rm(htrans, sum_htrans_id, ohe_htrans); invisible(gc())
 
#---------------------------
cat("Preprocessing new transactions...\n")

ntrans = new_merchant_trnx
ntrans = ntrans%>%
  mutate(purchase_amount = round(purchase_amount/ 0.00150265118 + 497.06,8))


##ntrans <- read_csv("../input/new_merchant_transactions.csv") %>% 
  left_join(read_csv("../input/merchants.csv"),
            by = "merchant_id", suffix = c("", "_y")) %>%
  select(-authorized_flag) %>% 
  rename(card = card_id)

ntrans <- new_merchant_trnx %>% 
  left_join(merchants,
            by = "merchant_id", suffix = c("", "_y")) %>%
  select(-authorized_flag) %>% 
  rename(card = card_id)


sum_ntrans_id <- ntrans %>%
  group_by(card) %>%
  summarise_at(vars(contains("_id")), n_distinct, na.rm = TRUE) 

ohe_ntrans <- ntrans %>%
  select(starts_with("category"), starts_with("most_recent")) %>% 
  mutate_all(factor) %>% 
  model.matrix.lm(~ . - 1, ., na.action = NULL) %>% 
  as_tibble()

fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
sum_ntrans <- ntrans %>%
  select(-starts_with("category"), -starts_with("most_recent"), -contains("_id")) %>% 
  add_count(card) %>%
  group_by(card) %>%
  mutate(date_diff = as.integer(diff(range(purchase_date))),
         prop = n() / sum(n)) %>% 
  ungroup() %>% 
  mutate(year = year(purchase_date),
         month = month(purchase_date),
         day = day(purchase_date),
         hour = hour(purchase_date),
         month_diff = as.integer(ymd("2018-12-01") - date(purchase_date)) / 30 + month_lag) %>% 
  select(-purchase_date) %>% 
  bind_cols(ohe_ntrans) %>% 
  group_by(card) %>%
  summarise_all(fn) %>% 
  left_join(sum_ntrans_id)

rm(ntrans, sum_ntrans_id, ohe_ntrans, fn); invisible(gc())

#---------------------------
cat("Joining datasets...\n")

## Adding RAddar's changed variable into train data


tr <- train 
te <- test

tri <- 1:nrow(tr)
y <- tr$target

tr_te <- tr %>% 
  select(-target) %>% 
  bind_rows(te) %>%
  rename(card = card_id) %>% 
  mutate(first_active_month = ymd(first_active_month, truncated = 1),
         year = year(first_active_month),
         month = month(first_active_month),
         date_diff = as.integer(ymd("2018-02-01") - first_active_month)) %>% 
  select(-first_active_month) %>% 
  left_join(sum_htrans, by = "card") %>% 
  left_join(sum_ntrans, by = "card") %>% 
  select(-card) %>% 
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) %>% 
  select_if(~ n_distinct(.x) > 1) %>% 
  data.matrix()

fin_v1 = as.data.frame(tr_te)
#write_csv(fin_v1, "FinalPreppedData.csv")
rm(tr, te, sum_htrans, sum_ntrans); invisible(gc())

#---------------------------
cat("Preparing data...\n")

tr_te = as.matrix(train1)

val <- caret::createDataPartition(y, p = 0.2, list = FALSE)
dtrain <- xgb.DMatrix(data = tr_te[tri, ][-val, ], label = y[-val])
dval <- xgb.DMatrix(data = tr_te[tri, ][val, ], label = y[val])
dtest <- xgb.DMatrix(data = tr_te[-tri, ])
cols <- colnames(tr_te)

rm(tr_te, y, tri); gc()

#---------------------------
cat("Training model...\n")
p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 5,
          eta = 0.1,
          max_depth = 10,
          min_child_weight = 100,
          gamma = 0,
          subsample = 0.9,
          colsample_bytree = 0.8,
          colsample_bylevel = 0.8,
          alpha = 0,
          lambda = 1)

set.seed(1)
m_xgb <- xgb.train(p, dtrain, 2000, list(val = dval), print_every_n = 20, early_stopping_rounds = 100)


xgb.importance(cols, model = m_xgb) %>% 
  xgb.plot.importance(top_n = 20) + theme_minimal()

#xgb.plot.importance(importance_matrix = importanceRaw)

#---------------------------
read_csv("../input/sample_submission.csv") %>%  
  mutate(target = predict(m_xgb, dtest)) %>%
  write_csv(paste0("tidy_elo_", round(m_xgb$best_score, 5), ".csv"))

sample %>%  
  mutate(target = predict(m_xgb, dtest)) %>%
  write_csv(paste0("tidy_elo_", round(m_xgb$best_score, 8), ".csv"))

#####################3

a = xgb.importance(cols, model = m_xgb)[1:40]
a$Feature

train1 = fin_v1%>%
  select(a$Feature)

fin_v1%>%
  select(va)



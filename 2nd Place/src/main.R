Sys.setenv(TZ="Europe/Paris") # Strange warning on my R versions - should update

library(lightgbm)
library(data.table)
library(RcppRoll)
library(dtplyr)
library(stringr)
library(tidyverse)
library(lubridate)

source("src/utilities.R")

SAMPLING_SIZE = 20 # Size of sampling - used in function generate_target(...)

###
# Read data ---------------------------------------------------------------
print("Read data")

holidays = read_csv("./data/holidays.csv") %>% select(-1)
metadata = read_csv("./data/metadata.csv")
submission_format = read_csv("./data/submission_format.csv")
# submission_frequency = read_csv("./data/submission_frequency.csv")
train = read_csv("./data/train.csv")
weather = read_csv("./data/weather.csv") %>% select(-1)

test = submission_format %>% 
  mutate(Value = NA)

###
# Feature engineering -----------------------------------------------------

# Dedup weather (SiteId, Timestamp) ---------------------------------------
print("Dedup weather (SiteId, Timestamp)")
# Dedup (SiteId, Timestamp) : Multiple weather station by SiteId, therefore we'll only keep the closest one
weather = weather %>% 
  group_by(SiteId, Timestamp) %>% 
  arrange(Distance) %>% 
  slice(1) %>% 
  ungroup()

weather = weather %>% 
  rename(Timestamp_weather = Timestamp) %>% 
  mutate(Timestamp_lead = Timestamp_weather)

weather = weather %>% as.data.table()

# Clean outliers in train data --------------------------------------------
print("Clean outliers in train data")
k = 10
train = train %>%
  group_by(ForecastId) %>%
  mutate(value_mean = mean(Value, na.rm = T)) %>% 
  ungroup() %>%
  mutate(Value = if_else((Value / value_mean) > k, # | (Value / value_roll_median) < 1/k,
                         true = as.double(NA), false = Value)) %>% 
  select(-value_mean)
rm(k)

# forecast_info - information about ForecastId ----------------------------
print("forecast_info - information about ForecastId")
forecast_info = test %>% 
  group_by(ForecastId) %>%
  mutate(timestep_minutes = (Timestamp - lag(Timestamp)) / dseconds(60)) %>% 
  summarise(size = n(), timestep_minutes = max(timestep_minutes, na.rm = T)) %>% 
  ungroup()

forecast_info = forecast_info %>% 
  mutate(step_day = 1440 / timestep_minutes,
         step_hour = 60 / timestep_minutes) %>% 
  mutate(step_hour = ifelse(step_hour < 1, NA, step_hour) %>% as.integer())

forecast_info_tmp = forecast_info %>%
  group_by(timestep_minutes) %>%
  arrange(size) %>%
  mutate(size_adjusted = last(size)) %>%
  ungroup() %>%
  arrange(ForecastId)

# fe_holidays -------------------------------------------------------------
fe_holidays = holidays %>% 
  mutate(Holiday = 1) %>% 
  distinct() %>% # Multiple holidays on same day...
  complete(Date = seq.Date(from = min(holidays$Date), to = max(holidays$Date), by = "days"), 
           SiteId, fill = list(Holiday = 0))

fe_holidays = fe_holidays %>% 
  group_by(SiteId) %>% 
  mutate(holiday_lag = lag(Holiday),
         holiday_lead = lead(Holiday)) %>% 
  ungroup()

fe_holidays = fe_holidays %>% as.data.table()
rm(holidays)

# fe_site - Feature engineering on SiteId ---------------------------------
print("fe_site - Feature engineering on SiteId")
fe_site = metadata %>%
  gather(key = dow, value = closed, -SiteId, -Surface, -Sampling, -BaseTemperature) %>% 
  mutate(dow = case_when(dow == "MondayIsDayOff" ~ 1,
                         dow == "TuesdayIsDayOff" ~ 2,
                         dow == "WednesdayIsDayOff" ~ 3,
                         dow == "ThursdayIsDayOff" ~ 4,
                         dow == "FridayIsDayOff" ~ 5,
                         dow == "SaturdayIsDayOff" ~ 6,
                         TRUE ~ 7))

fe_site = fe_site %>% 
  mutate(closed = closed %>% as.logical() %>% as.integer())

#dow_holiday - first dow closed
fe_site_dow_holiday = fe_site %>% 
  group_by(SiteId) %>% 
  filter(lag(closed) == 0, closed == 1) %>% 
  ungroup() %>% 
  rename(dow_holiday = dow) %>% 
  mutate(dow_lag_holiday = dow_holiday - 1) %>% #dow_lag_holiday - first dow before closed
  select(SiteId, dow_lag_holiday, dow_holiday)

#dow_lead_holiday - next dow after closed
fe_site_dow_holiday = fe_site %>% 
  bind_rows(fe_site %>% filter(dow == 1)) %>% 
  group_by(SiteId) %>% 
  filter(lead(closed) == 0, closed == 1) %>% 
  ungroup() %>% 
  rename(dow_lead_holiday = dow) %>%
  mutate(dow_lead_holiday = (dow_lead_holiday + 1)) %>% 
  mutate(dow_lead_holiday = ifelse(dow_lead_holiday == 8, 1, dow_lead_holiday)) %>% # Yes checked
  select(SiteId, dow_lead_holiday) %>% 
  inner_join(fe_site_dow_holiday, ., by = "SiteId")

fe_site = fe_site %>% 
  left_join(fe_site_dow_holiday, by = "SiteId")
rm(fe_site_dow_holiday)

fe_site = fe_site  %>% as.data.table()

# data - info about Timestamp ---------------------------------------------
# data is our backbone data where we'll extract values from
data = train %>% 
  select(SiteId, ForecastId, Timestamp, Value) %>% 
  as.data.table()

setkeyv(weather, c("SiteId", "Timestamp_lead"))
setkeyv(data, c("SiteId", "Timestamp"))

data = weather[data, roll="nearest"]

data %>% setnames("Timestamp_lead", "Timestamp")
data[ , c("Timestamp_weather", "Distance") := NULL]

# fe_backward_last : feature engineering back in the time ----------------------
print("fe_backward_last : feature engineering back in the time")
fe_backward_last = data %>% 
  select(SiteId, ForecastId, Timestamp, Value, Temperature) %>% as.data.table()

fe_backward_last[ , c("value_last_mean_60", "value_last_mean_192", 
                      "value_last_sd_60", "value_last_sd_192",
                      "temperature_last_mean_60", "temperature_last_mean_192") :=
                        list(roll_mean(Value, n = 60, align = "right", fill = NA, na.rm = T),
                             roll_mean(Value, n = 192, align = "right", fill = NA, na.rm = T),
                             roll_sd(Value, n = 60, align = "right", fill = NA, na.rm = T),
                             roll_sd(Value, n = 192, align = "right", fill = NA, na.rm = T),
                             roll_mean(Temperature, n = 60, align = "right", fill = NA, na.rm = T),
                             roll_mean(Temperature, n = 192, align = "right", fill = NA, na.rm = T)),
                      by = c("ForecastId")]

fe_backward_last[, c("Value", "Temperature") := NULL]

fe_backward_last = merge(fe_backward_last, 
      forecast_info_tmp %>% select(ForecastId, size_adjusted) %>% as.data.table(), 
      by = c("ForecastId"))
rm(forecast_info_tmp)

fe_backward_last = fe_backward_last %>% 
  mutate(value_last_mean = ifelse(size_adjusted == 60, value_last_mean_60, value_last_mean_192),
         value_last_sd = ifelse(size_adjusted == 60, value_last_sd_60, value_last_sd_192),
         temperature_last_mean = ifelse(size_adjusted == 60, temperature_last_mean_60, temperature_last_mean_192)) %>% 
  select(-ends_with("60"), -ends_with("192"), -size_adjusted)

fe_backward_last = fe_backward_last %>% filter(!is.na(value_last_mean))

# fe_backward_tmp ------------------------------------------------------------
print("fe_backward_tmp")
# FE on same time, dow, ...

setkeyv(data, cols = c("ForecastId"))
fe_backward_tmp = data %>% 
  merge(forecast_info %>% as.data.table(), by = "ForecastId")

# Consolidate value by averaging backward the last K values (eg 10:00, 10:15, 10:30, 10:45) 
fe_backward_tmp = fe_backward_tmp %>% 
  .[, time := strftime(Timestamp, format="%H:%M:%S")] %>% 
  .[, value_4 := roll_mean(Value, n = 4, align = "right", fill = NA, na.rm = T), 
    by = c("ForecastId")]

# fe_backward_last_day # fe_backward_last_day (and time) -------------------
print("fe_backward_last_day")
fe_backward_last_day = fe_backward_tmp %>% copy()

fe_backward_last_day[ , c("value_time_mean_7", "value_time_mean_3", 
                          "temperature_time_mean_7", "temperature_time_mean_3", 
                          "value_4_time_mean_7", "value_4_time_mean_3") :=
                        list(roll_mean(Value, n = 7, align = "right", fill = NA, na.rm = T),
                             roll_mean(Value, n = 3, align = "right", fill = NA, na.rm = T),
                             roll_mean(Temperature, n = 7, align = "right", fill = NA, na.rm = T),
                             roll_mean(Temperature, n = 3, align = "right", fill = NA, na.rm = T),
                             roll_mean(value_4, n = 7, align = "right", fill = NA, na.rm = T),
                             roll_mean(value_4, n = 3, align = "right", fill = NA, na.rm = T)),
                      by = c("ForecastId", "time")] %>% 
  setnames(c("Value", "Temperature", "value_4", "Timestamp"), 
           c("value_time_mean_1", "temperature_time_mean_1", "value_4_time_mean_1", "Timestamp_last_day")) %>% 
  .[, c("time", "size", "timestep_minutes", "step_day", "step_hour") := NULL]

# fe_backward_last_dow ----------------------------------------------------
print("fe_backward_last_dow")
fe_backward_last_dow = fe_backward_tmp %>% copy()

fe_backward_last_dow %>% 
  .[ , dow := wday(Timestamp, week_start = getOption("lubridate.week.start", 1))] %>% 
  .[ , c("value_dow_mean_3", "value_dow_mean_2", "temperature_dow_mean_3", "temperature_dow_mean_2") := 
       list(roll_mean(Value, n = 3, align = "right", fill = NA, na.rm = T),
            roll_mean(Value, n = 2, align = "right", fill = NA, na.rm = T),
            roll_mean(Temperature, n = 3, align = "right", fill = NA, na.rm = T),
            roll_mean(Temperature, n = 2, align = "right", fill = NA, na.rm = T)), 
     by = c("ForecastId", "dow", "time")] %>% 
  setnames(c("Value", "Timestamp", "Temperature"), 
           c("value_dow_mean_1", "Timestamp_last_dow", "temperature_dow_mean_1")) %>% 
  .[, c("time", "size", "timestep_minutes", "step_day", "step_hour", "dow") := NULL]

# fe_backward_last_closed -------------------------------------------------
print("fe_backward_last_closed")
# FE on last opened/closed day
fe_backward_last_closed = fe_backward_tmp %>% copy()
fe_backward_last_closed[ , dow := wday(Timestamp, week_start = getOption("lubridate.week.start", 1))]

fe_backward_last_closed = fe_site %>% 
  select(SiteId, dow, closed) %>% as.data.table() %>% 
  merge(fe_backward_last_closed, ., by = c("SiteId", "dow"))

fe_backward_last_closed %>% 
  setnames(c("Value", "Temperature", "value_4", "Timestamp"), 
           c("value_closed_mean_1", "temperature_closed_mean_1", "value_4_closed_mean_1", "Timestamp_last_closed")) %>% 
  .[, c("dow", "size", "timestep_minutes", "step_day", "step_hour", "time") := NULL]

rm(fe_backward_tmp)

# df_target - Generate target values - Y ----------------------------------------------
print("df_target - Generate target values - Y")

target = train %>% 
  select(SiteId, ForecastId, Timestamp, Value) %>% 
  tbl_dt()

# Don't generate target which were filtered in fe_backward_last because we can't have value_last_mean
target = target %>% 
  merge(fe_backward_last[ , .(ForecastId, SiteId, Timestamp)], 
        by = c("ForecastId", "SiteId", "Timestamp"))

# Don't generate when all Value are NA
target = target %>% # Ugly - Learn how to do it in data.table way ...
  group_by(ForecastId) %>% 
  filter(sum(Value, na.rm = T) > 0) %>% 
  ungroup()

# Don't process empty training sets in generate_target_loop
forecast_info = forecast_info %>%
  inner_join(target %>% distinct(ForecastId), by = "ForecastId")

system.time(df_target <- generate_target_loop(target, forecast_info$ForecastId, 
                                              forecast_info$size, SAMPLING_SIZE = SAMPLING_SIZE))
rm(target)

df_target_test = generate_target_test(train, submission_format)

df_target = df_target %>% mutate(set = "train") %>% 
  bind_rows(df_target_test %>% mutate(set = "test"), .)
rm(df_target_test)

df_target = df_target %>% 
  mutate(time = round(hour(Timestamp_lead) + minute(Timestamp_lead) / 60, 1)) %>% 
  mutate(dow = wday(Timestamp_lead, week_start = getOption("lubridate.week.start", 1)))

# Slightly better in local - slightly worse in public leaderboard - think it's help 
df_target = df_target %>% # Circular data
  mutate(time2 = (time + 12) %% 24)

# Compute last diverse Timestamp available : Timestamp_last_day & Timestamp_last_dow
df_target = forecast_info %>% 
  select(ForecastId, timestep_minutes, step_day) %>% 
  as.data.table() %>% 
  merge(df_target, by = "ForecastId") 

df_target = df_target %>% 
  mutate(Timestamp_last_day = Timestamp_lead - dminutes(timestep_minutes) * 
           ceiling(lead / step_day) * step_day) %>% 
  mutate(Timestamp_last_dow = Timestamp_lead - dminutes(timestep_minutes) *
           ceiling(lead / (step_day*7)) * step_day * 7) %>%
  select(-timestep_minutes, -step_day)

# Might be somehow usefull
df_target = df_target %>%
  mutate(month = month(Timestamp_lead),
         #year = year(Timestamp_lead),
         day = day(Timestamp_lead))

# Generate learning set ---------------------------------------------------
print("Generate learning set")
df_learning = df_target %>% copy()

print("Join fe_backward_last and df_learning")
setkeyv(fe_backward_last, c("SiteId", "ForecastId", "Timestamp"))
setkeyv(df_learning, c("SiteId", "ForecastId", "Timestamp"))
df_learning = merge(df_learning, fe_backward_last)
print(df_learning %>% dim())

print("Join fe_backward_last_day")
setkeyv(df_learning, c("SiteId", "ForecastId", "Timestamp_last_day"))
setkeyv(fe_backward_last_day, c("SiteId", "ForecastId", "Timestamp_last_day"))
df_learning = merge(df_learning, fe_backward_last_day, all.x=TRUE) 
print(df_learning %>% dim())

print("Join fe_backward_last_dow")
setkeyv(df_learning, c("SiteId", "ForecastId", "Timestamp_last_dow"))
setkeyv(fe_backward_last_dow, c("SiteId", "ForecastId", "Timestamp_last_dow"))
df_learning = merge(df_learning, fe_backward_last_dow, all.x=TRUE) 
print(df_learning %>% dim())
df_learning[ , distance2last_dow := (Timestamp_lead - Timestamp_last_dow) / ddays(1)]

print("Join fe_site and df_learning")
setkeyv(df_learning, c("SiteId", "dow"))
df_learning = fe_site %>%
  merge(df_learning, ., by = c("SiteId", "dow"))
print(df_learning %>% dim())

print("Join fe_holidays and df_learning")
df_learning[ , Date := date(Timestamp_lead)]

setkeyv(df_learning, c("SiteId", "Date"))
df_learning = merge(df_learning, fe_holidays, by =  c("SiteId", "Date"), all.x = T)
print(df_learning %>% dim())
df_learning[ , Date := NULL]

# To make the code cleaner, this FE should be done above I think
df_learning[ , Holiday := ifelse(is.na(Holiday), 0, Holiday)]
df_learning[ , holiday_lead := ifelse(is.na(holiday_lead), 0, holiday_lead)]
df_learning[ , holiday_lag := ifelse(is.na(holiday_lag), 0, holiday_lag)]

# We don't know if Holidays are observed or not in Site, don't know if it helps or not
# Therefore I choose to be more conversative, create new variables and not directly replace (dow, closed) variables 
# and therefore how joins are done on this variable further in the code
# If Holiday => then assign as closed day
df_learning[ , Holiday := ifelse(Holiday == 1 | closed == 1, 1, 0)]
# If next day is holiday and today is not closed, dow is equal to dow_lag_holiday 
df_learning[ , dow_lag_holiday := ifelse(holiday_lead == 1 & closed == 0, dow_lag_holiday, dow)]
# If yesterday is holiday and today is not closed, dow is equal to assign dow_lead_holiday
df_learning[ , dow_lead_holiday := ifelse(holiday_lag == 1 & closed == 0, dow_lead_holiday, dow)]

print("Join fe_backward_last_closed")
# Timestamp_last_closed is a rolling join on c("SiteId", "ForecastId", "time", "closed", "Timestamp_last_day")
df_learning[, time := strftime(Timestamp_lead, format="%H:%M:%S")]

fe_backward_last_closed[, time := strftime(Timestamp_last_closed, format="%H:%M:%S")]
fe_backward_last_closed[ , Timestamp_last_day := Timestamp_last_closed]

setkeyv(fe_backward_last_closed, c("SiteId", "ForecastId", "time", "closed", "Timestamp_last_day"))
setkeyv(df_learning, c("SiteId", "ForecastId", "time", "closed", "Timestamp_last_day"))
df_learning = fe_backward_last_closed[df_learning, roll=T]
print(df_learning %>% dim())

df_learning[ , distance2last_closed := (Timestamp_lead - Timestamp_last_closed) / ddays(1)]

# Do it above with another name like time_numeric then delete time because it's a character string
df_learning[ , time := round(hour(Timestamp_lead) + minute(Timestamp_lead) / 60, 1)] # Add back time for model

print("Join weather and df_learning")
weather = weather %>% as.data.table()

setkeyv(weather, c("SiteId", "Timestamp_lead"))
setkeyv(df_learning, c("SiteId", "Timestamp_lead"))
df_learning = weather[df_learning, roll="nearest"]
print(df_learning %>% dim())
df_learning[ , time2weather := (Timestamp_lead - Timestamp_weather) / dseconds(1)]

print("FE / Normalization between Temperature from Timestamp_lead & Temperature from Timestamp_XXX")
df_learning[ , temperature_Temperature_mean_rt := Temperature / temperature_last_mean]
df_learning[ , temperature_Temperature_mean_diff := Temperature - temperature_last_mean]
# To make it more readable should be in the opposite order IMO - but has same effect
df_learning[ , temperature_time_mean_1_Temperature_diff := temperature_time_mean_1 - Temperature]
df_learning[ , temperature_dow_mean_1_Temperature_diff := temperature_dow_mean_1 - Temperature]
df_learning[ , temperature_closed_mean_1_Temperature_diff := temperature_closed_mean_1 - Temperature]

print("Normalize all columns related to values here")
df_learning[ , Value_lead := Value_lead / value_last_mean]
df_learning[ , value_closed_mean_1 := value_closed_mean_1 / value_last_mean]
df_learning[ , value_4_closed_mean_1 := value_4_closed_mean_1 / value_last_mean]
df_learning[ , value_time_mean_1 := value_time_mean_1 / value_last_mean]
df_learning[ , value_4_time_mean_1 := value_4_time_mean_1 / value_last_mean]
df_learning[ , value_time_mean_7 := value_time_mean_7 / value_last_mean]
df_learning[ , value_time_mean_3 := value_time_mean_3 / value_last_mean]
df_learning[ , value_4_time_mean_7 := value_4_time_mean_7 / value_last_mean]
df_learning[ , value_4_time_mean_3 := value_4_time_mean_3 / value_last_mean]
df_learning[ , value_4 := value_4 / value_last_mean]
df_learning[ , value_dow_mean_3 := value_dow_mean_2 / value_last_mean]

print("Join timestep_minutes and df_learning")
df_learning = merge(df_learning, forecast_info %>% 
                      select(ForecastId, size, timestep_minutes) %>% as.data.table(),
                    by = c("ForecastId"))

# Breaks otherwise in train_model_multiple because of row_number() - it's ugly
# See https://stackoverflow.com/questions/12925063/numbering-rows-within-groups-in-a-data-frame?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
df_learning = df_learning %>%
  as_tibble() # Naughty

print(df_learning %>% dim())

# Delete 481 points on SiteID == 19 because division by 0 otherwise
# df_learning %>% filter(SiteId == 19) %>% distinct(Timestamp, value_last_mean) %>% ggplot(aes(Timestamp, value_last_mean)) + geom_point()
df_learning = df_learning %>% filter(value_last_mean > 0)

# Modeling -------------------------------------------------------------------
print("Modeling")
system.time(
  res_train_model_hierarchical <- df_learning %>% 
    # filter(SiteId <= 50) %>%
    train_model_hierarchical_multiple(train_full = F)
    # train_model_hierarchical(train_full = F, verbose = -1)
)

submission = res_train_model_hierarchical$submission
df_valid_pred = res_train_model_hierarchical$df_valid_pred
df_train_pred = res_train_model_hierarchical$df_train_pred
rm(res_train_model_hierarchical)

df_valid_pred %>% 
  mutate(Value_lead_pred = Value_lead_pred_4) %>% 
  compute_loss()

# Generate submission.csv -------------------------------------------------
# Correct missing submission
submission = submission_format %>% 
  select(-Value) %>% 
  left_join(submission %>% select(ForecastId, Timestamp_lead, Value), 
            by = c("ForecastId" = "ForecastId", "Timestamp" = "Timestamp_lead"))

correct_pb_pred = submission %>% 
  filter(is.na(Value)) %>%
  distinct(SiteId, ForecastId) %>% 
  rowwise() %>% # Not sure it's the best practice see here https://rpubs.com/wch/200398
  mutate(value_mean_before = get_value_mean_before(v_ForecastId = ForecastId, v_SiteId = SiteId)) %>% 
  ungroup()

submission = submission %>% 
  left_join(correct_pb_pred, by = c("SiteId", "ForecastId")) %>% 
  mutate(Value = if_else(is.na(value_mean_before), Value, value_mean_before)) %>% 
  select(-value_mean_before)
rm(correct_pb_pred)

sys_time = Sys.time() %>% str_replace_all(" |:", "-")
file_name = paste0("./output/submission_",
                   sys_time, ".csv")

submission %>% 
  write_csv(file_name)

system(paste0("zip ", file_name, ".zip ", file_name))
system(paste0("rm ", file_name))




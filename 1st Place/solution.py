import preprocess
import lgb


preprocess.main(update_means_only = False, forced_update = False) # preprocessing data
lgb.main()  # train model and predict
print('done.')

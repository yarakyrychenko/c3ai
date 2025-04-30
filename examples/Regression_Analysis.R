library(lme4)

data <- read.csv("principle_preferences.csv")

# overall model for dimensions/factors
set.seed(1)
all_model_dimension <- glmer(same_choice_probs ~ -1 + dimension  + (1 | prompt_id_full) +  (1 | principle_id), 
                             data = data,
                             family = binomial)

tabdim <- tab_model(all_model_dimension, title = "", 
                    digits.p = 4)
tabdim

# overall model for framing analysis
set.seed(1)
all_model_framing <- glmer(same_choice_probs ~ pos_neg_framing + trait_beh_framing + (1 | prompt_id_full) +  (1 | principle_id), 
                      data = data,
                      family = binomial)

tabframing <- tab_model(all_model_framing, title = "", 
                   digits.p = 4)
tabframing


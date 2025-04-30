library(EGAnet)
library(tidyverse)

data <- read.csv("principle_preferences.csv")

dat <- data[c('prompt_id_full','principle_id','same_choice_probs')] %>%
  pivot_wider(names_from = principle_id, values_from = same_choice_probs)
vars <- colnames(dat)[2:length(dat)]
vars

new_vars <- data %>%
  distinct(principle_id, name)
indices <- match(vars, names(dat))
names(dat)[indices] <- new_vars$name
old_vars <- vars
vars <- new_vars$name


data_st <- data[data$data_category=="Harmlessness",]
dat_harmless <- data_st[c('prompt_id_full','principle_id','same_choice_probs')] %>%
  pivot_wider(names_from = principle_id, values_from = same_choice_probs)
names(dat_harmless)[indices] <- new_vars$name

data_st <- data[data$data_category=="Helpfulness",]
dat_helpful <- data_st[c('prompt_id_full','principle_id','same_choice_probs')] %>%
  pivot_wider(names_from = principle_id, values_from = same_choice_probs)
names(dat_helpful)[indices] <- new_vars$name

data_st <- data[data$data_category=="General",]
dat_general <- data_st[c('prompt_id_full','principle_id','same_choice_probs')] %>%
  pivot_wider(names_from = principle_id, values_from = same_choice_probs)
names(dat_general)[indices] <- new_vars$name

uva_harmless <- UVA(
  data = dat_harmless[, vars],
  key = vars,
  cut.off = 0.25,
  seed = 123
)

boot_ega_harmless <- bootEGA(
  data = uva_harmless$reduced_data,
  seed = 1, 
  iter = 500
)

boot_ega_harmless_plot <- plot(boot_ega_harmless, edge.size = 6, node.size = colSums(boot_ega_harmless$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("boot_ega_harmless_uva.pdf", boot_ega_harmless_plot, width=24,height=18) 

export_graph <- function(boot, name){
  tg <- data.frame(boot[["typicalGraph"]][["graph"]])
  tg$rownames <- rownames(boot[["typicalGraph"]][["graph"]])
  tg$nodesize <- colSums(boot$typicalGraph$graph)^2 * 16
  write_csv(tg,name)
}

export_graph(boot_ega_harmless,"graph_boot_ega_harmless.csv")

uva_helpful <- UVA(
  data = dat_helpful[, vars],
  key = vars,
  cut.off = 0.25,
  seed = 123
)

boot_ega_helpful <- bootEGA(
  data = uva_helpful$reduced_data,
  seed = 1,  
  iter = 500
)

boot_ega_helpful_plot <- plot(boot_ega_helpful, edge.size = 6, node.size = colSums(boot_ega_helpful$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("boot_ega_helpful_uva.pdf", boot_ega_helpful_plot, width=24,height=18) 

export_graph(boot_ega_helpful,"graph_boot_ega_helpful.csv")

uva_general <- UVA(
  data = dat_general[, vars],
  key = vars,
  cut.off = 0.25,
  seed = 123
)

boot_ega_general <- bootEGA(
  data = uva_general$reduced_data,
  seed = 1,  
  iter = 500
)

boot_ega_general_plot <- plot(boot_ega_general, edge.size = 6, node.size = colSums(boot_ega_general$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("boot_ega_general_uva.pdf", boot_ega_general_plot, width=24,height=18) 

export_graph(boot_ega_general,"graph_boot_ega_general.csv")

generate_netloadings <- function(uva, boot_ega, dat, vars, stability_cutoff){
  
  # Step 1: Compute dimension stability
  stability_reduced <- dimensionStability(boot_ega)
  
  # Step 2: Identify stable items
  stable_items <- names(stability_reduced$item.stability$item.stability$empirical.dimensions[
    stability_reduced$item.stability$item.stability$empirical.dimensions > stability_cutoff
  ])
  
  # Step 4: Identify super stable items
  suber_stable_items <- intersect(stable_items, colnames(uva$reduced_data))
  
  # Step 5: Perform EGA on the super stable items
  ega_super_stable <- EGA(data = dat[, suber_stable_items])
  
  # Step 6: Compute network loadings
  netloads.stable <- net.loads(
    data = dat[, suber_stable_items],
    wc = ega_super_stable$wc,
    A = ega_super_stable$network,
    loading.method = "experimental"
  )
  
  return(netloads.stable)
}

get_non_na_rows <- function(mat) {
  non_na_rows <- apply(mat, 1, function(row) any(!is.na(row)))
  mat[non_na_rows, , drop = FALSE]
}

net_threshold <- .9
threshold <- .25 #0.35
#### Harmless ####
mat_harmless <- generate_netloadings(uva_harmless,boot_ega_harmless,dat_harmless,vars,net_threshold)$std
filtered_mat_harmless <- ifelse(mat_harmless > threshold, mat_harmless, NA)
nonna_harmless <- get_non_na_rows(filtered_mat_harmless)
nrow(nonna_harmless)
rownames(nonna_harmless)

#### Helpful ####
mat_helpful <- generate_netloadings(uva_helpful,boot_ega_helpful,dat_helpful,vars,net_threshold)$std
filtered_mat_helpful <- ifelse(mat_helpful  > threshold, mat_helpful, NA)
nonna_helpful <- get_non_na_rows(filtered_mat_helpful)
nrow(nonna_helpful)
rownames(nonna_helpful)
intersect(rownames(nonna_helpful),rownames(nonna_harmless))

#### General ####
mat_general <- generate_netloadings(uva_general,boot_ega_general,dat_general,vars,.9)$std
filtered_mat_general <- ifelse(mat_general  > .2, mat_general, NA)
nonna_general<- get_non_na_rows(filtered_mat_general)
nrow(nonna_general)
rownames(nonna_general)
gh <- intersect(rownames(nonna_general),rownames(nonna_harmless))
ghel <- intersect(rownames(nonna_general),rownames(nonna_helpful))

alldiff <- intersect(gh,ghel)

allu <- union(union(rownames(nonna_general),rownames(nonna_harmless)),rownames(nonna_helpful))

#### OVERALL ####
uva_all<- UVA(
  data = dat[, vars],
  key = vars,
  cut.off = 0.25,
  seed = 123
)

boot_ega_uva <- bootEGA(
  data = uva_all$reduced_data,
  seed = 1, 
  iter = 500
)


boot_ega_plot <- plot(boot_ega_uva, edge.size = 6, node.size = colSums(boot_ega_uva$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("boot_ega_all_new_uva.pdf", boot_ega_plot, width=24,height=18) 

export_graph(boot_ega_uva,"graph_boot_ega_overall_uva.csv")

mat_all <- generate_netloadings(uva_all,boot_ega_uva,dat,vars,net_threshold)$std
filtered_mat_all <- ifelse(mat_all  > threshold, mat_all, NA)
nonna_all<- get_non_na_rows(filtered_mat_all)
nrow(nonna_all)
rownames(nonna_all)

dims <- boot_ega_uva[["EGA"]][["dim.variables"]]
data <- merge(data, data.frame(dims), by.x= "name", by.y = "items", all.x = TRUE)

### EGA ALL NO UVA ###


boot_ega<- bootEGA(
  data = dat[, vars],
  seed = 1, 
  iter = 500
)


boot_ega_plot <- plot(boot_ega, edge.size = 6, node.size = colSums(boot_ega$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("boot_ega_all_new_all.pdf", boot_ega_plot, width=24,height=18) 


dims <- boot_ega[["EGA"]][["dim.variables"]]
data <- merge(data, data.frame(dims), by.x= "name", by.y = "items", all.x = TRUE)

#### HARMLESSNESS ONLY HARMLESS DATA AND ANTHROPIC ####
data_st <- data[data$dataset=="Harmlessness (HH-RLHF)" & data$principle_category=="Anthropic",]
dat_hh <- data_st[c('prompt_id_full','principle_id','same_choice_probs')] %>%
  pivot_wider(names_from = principle_id, values_from = same_choice_probs)
vars_hh <- colnames(dat_hh)[2:length(dat_hh)]
vars_hh


uva_hh <- UVA(
  data = dat_hh[,vars_hh],
  key = vars,
  cut.off = 0.25,
  seed = 123,
)

boot_ega_hh <- bootEGA(
  data = uva_hh$reduced_data,
  seed = 1,  
)

boot_ega_plot_hh <- plot(boot_ega_hh, edge.size = 6, node.size = colSums(boot_ega_hh$typicalGraph$graph)^2 * 16) + theme(legend.position = "bottom")
ggsave("boot_ega_all_new_uva_hh.pdf", boot_ega_plot_hh, width=24,height=18) 

nodesize <- as.data.frame(colSums(boot_ega_hh$typicalGraph$graph))
colnames(nodesize) <- c("nodesize")
nodesize$principle_id <- rownames(nodesize)
write_csv(nodesize, "hh_nodesize.csv")

mat_hh <- generate_netloadings(uva_hh,boot_ega_hh,dat_hh,vars_hh,.75)$std
filtered_mat_hh <- ifelse(mat_hh > .25, mat_hh, NA)
nonna_hh <- get_non_na_rows(filtered_mat_hh)
nrow(nonna_hh)
rownames(nonna_hh)


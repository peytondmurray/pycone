#### FORMAT R VALUES TO PLOT
df_R_melt <- melt(r_table, id.vars = c("sites"))
colnames(df_R_melt) <- c("site", "timesteps", "r")

# add species column
site_letters <- gsub("[[:digit:]]", "", df_R_melt$site)
species <- substr(site_letters, 1, nchar(site_letters) - 3)
df_R_melt$species <- species

# sites are the NAMES of the sites versus site is column on a table
sites <- rownames(r_table)
# scatter of r
first_day <- substr(toString(colnames(r_table)[1]), 1, 5)
r_title <- glue("r values
first day: {first_day} | duration: {duration} days | onset: {onset} days")

# PLOT R VALUES
# smoothing function
# https://www.sharpsightlabs.com/blog/geom_smooth/
#   DEFAULTS
#     loess function = local polynomial regression
#     span = 0.75.
#     level = 0.95, confidence interval around the line.

imgr_name <- glue("{newdir}/r plot__duration{duration} offset{i}.png")

p <- ggplot(df_R_melt, aes(timesteps, r^2)) +
  geom_point(size = 1, aes(color = site, alpha = r^2)) +
  geom_smooth(aes(x = as.numeric(timesteps), y = r, color = site, alpha = 0.1)) +
  facet_grid(~species) +
  ggtitle(r_title) +
  scale_x_discrete(breaks = seq(0, 30, 5)) +
  theme(legend.position = "none")
png(imgr_name)
print(p)
dev.off()

# imgr2_name = glue('{newdir}/r^2 plot__duration{duration} offset{i}.png')
q <- ggplot(df_R_melt, aes(timesteps, r)) +
  geom_point(size = 1, aes(color = species, alpha = r^2)) +
  geom_smooth(aes(x = as.numeric(timesteps), y = r, color = species, alpha = 0.1)) +
  facet_grid(~species) +
  ggtitle(r_title) +
  scale_x_discrete(breaks = seq(0, 30, 5)) +
  theme(legend.position = "none")
# png(imgr2_name)
# print(q)
# dev.off()


# EXTRA STUFF TO PLOT dT and cone
# plot dT
# wide -> long
#  http://www.cookbook-r.com/Manipulating_data/Converting_data_between_wide_and_long_format/

## Specify id.vars: the variables to keep but not split apart on
# melt(olddata_wide, id.vars=c("subject", "sex"))
exp_dT_melt <- melt(exp_dT, id.vars = c("dYear", "#timestep"))

# exp(dT)
ggplot(exp_dT_melt, aes(x = value, stat = "count", fill = variable)) +
  facet_wrap(as.factor(variable) ~ .) +
  geom_histogram(bins = 10) +
  ggtitle("exp(dT) distribution by site") +
  theme(legend.position = "none")

ggplot(exp_dT_melt, aes(x = value)) +
  geom_freqpoly(
    mapping = NULL,
    data = NULL,
    stat = "bin",
    position = "identity",
    na.rm = FALSE,
    show.legend = NA,
    inherit.aes = TRUE
  ) +
  facet_wrap(as.factor(variable) ~ .) +
  ggtitle(glue("exp(dT) distribution by site"))

# dT
ggplot(exp_dT_melt, aes(x = log(value), stat = "count", fill = variable)) +
  facet_wrap(as.factor(variable) ~ .) +
  geom_histogram(bins = 10) +
  ggtitle("dT distribution by site") +
  theme(legend.position = "none")

ggplot(exp_dT_melt, aes(x = log(value))) +
  geom_freqpoly(
    mapping = NULL,
    data = NULL,
    stat = "bin",
    position = "identity",
    na.rm = FALSE,
    show.legend = NA,
    inherit.aes = TRUE
  ) +
  facet_wrap(as.factor(variable) ~ .) +
  ggtitle(glue("dT distribution by site"))

# CONE
cone_melt <- melt(cone)
colnames(cone_melt) <- c("Year", "site", "value")
ggplot(cone_melt, aes(x = value, stat = "count", fill = site)) +
  facet_wrap(as.factor(site) ~ .) +
  geom_histogram(bins = 10) +
  ggtitle("Cone values by site") +
  theme(legend.position = "none")

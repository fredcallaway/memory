library(tidyverse)
library(lme4)
library(jtools)
library(magrittr)
library(purrr)
library(rmdformats) 
library(patchwork)
library(jsonlite)
library(tidyjson)
library(ggbeeswarm)
library(stickylabeller)
library(ggeffects)
library(rlang)
library(knitr)

options(
    "summ-model.info"=FALSE, 
    "summ-model.fit"=FALSE, 
    "summ-re.table"=FALSE, 
    "summ-groups.table"=FALSE,
    "jtools-digits"=3
)
WIDTH = 7.5; HEIGHT = 2.5

RED =  "#E41A1C" 
BLUE =  "#377EB8" 
GREEN =  "#4DAF4A" 
PURPLE =  "#984EA3" 
ORANGE =  "#FF7F00" 
YELLOW =  "#FFDD47" 
GRAY = "#8E8E8E"
BLACK = "#1B1B1B"

theme_set(theme_bw(base_size = 14))
theme_update(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank()
)

update_geom_defaults("line", list(size = 1.2))

kable = knitr::kable
glue = glue::glue

response_type_colors = scale_colour_manual(values=c(
    GREEN,
    ORANGE,
    RED,
    GRAY,
    BLACK
), aesthetics=c("fill", "colour"), name="Response Type")

word_type_colors = scale_colour_manual(values=c(
    BLUE,
    YELLOW
), aesthetics=c("fill", "colour"), name="Word Memorability")

json_to_columns <- function(df, column){
    json_df = df %>% 
        pull({{column}}) %>% 
        spread_all %>%
        as_tibble %>%
        select(-document.id)
    df %>% 
        select(-{{column}}) %>% 
        bind_cols(json_df)
}

zscore = function(x) as.vector(scale(x))

system('mkdir -p figs')
system('mkdir -p .fighist')
fig = function(name="tmp", w=4, h=4, dpi=320, ...) {
    if (isTRUE(getOption('knitr.in.progress'))) {
        show(last_plot())
        return()
    }
    ggsave("/tmp/fig.png", width=w, height=h, dpi=dpi, ...)
    stamp = format(Sys.time(), "%m-%d-%H-%M-%S")
    p = glue('.fighist/{name}-{stamp}.png')
    system(glue('mv /tmp/fig.png {p}'))
    system(glue('cp {p} figs/{name}.png'))
    # invisible(dev.off())
    # knitr::include_graphics(p)
}

smart_print = function(x, ...) {
    if (isTRUE(getOption('knitr.in.progress'))) {
        cat(knit_print(x, ...))
    } else {
        print(x, ...)
    }
}

inject = rlang::inject
tidylmer = function(data, xvar, yvar) {
    y = ensym(yvar)
    x = ensym(xvar)
    inject(lmer(!!y ~ !!x + (!!x | wid), data=data))
}
tidylm = function(data, xvar, yvar) {
    y = ensym(yvar)
    x = ensym(xvar)
    inject(lm(!!y ~ !!x, data=data))
}

pretty_names = list(
    strength_first = "First Cue Memory Strength",
    strength_second = "Second Cue Memory Strength",
    rel_strength = "Relative Memory Strength",
    prop_first = "Proportion Fixate First",
    rt = "Reaction Time",
    last_strength = "Last Fixated Strength"
)


def_breaks = list(
    strength_first = seq(-3.5, 3.5),
    rel_strength = seq(-4.9, 4.9, 1.4)
)


pretty_name = function(x) {
    if (!is_null(pretty_names[[x]])) return(pretty_names[[x]])
    x %>% 
        str_replace_all("_", " ") %>% 
        str_replace_all("pres", "fixation") %>% 
        str_to_title
}

# %% --------

make_breaks = function(x, n=7, q=.025) {
    xmin = quantile(x, q, na.rm=T)
    xmax = quantile(x, 1 - q, na.rm=T)
    seq(xmin, xmax, length.out=n)
}

regress = function(data, xvar, yvar, bins=6, bin_range=0.95, mixed=TRUE) {
    x = ensym(xvar); y = ensym(yvar)
    preds = data %>% 
        group_by(name) %>% 
        group_modify(function(data, grp) {
            model = if (grp$name == "Human") {
                if (mixed) {
                    model = inject(lmer(!!y ~ !!x + (!!x | wid), data=data))
                } else {
                    model = inject(lm(!!y ~ !!x, data=data))
                }
                print(glue("N = {nrow(data)}"))
                smart_print(summ(model))
                tibble(ggpredict(model, terms = glue("{x} [n=30]}")))
            } else {
                model = inject(lm(!!y ~ !!x, data=data))
                tibble(ggpredict(model, terms = glue("{x} [n=30]}")))
            }
        })

    xx = filter(data, name == "Human")[[x]]
    q = (1 - bin_range) / 2
    xmin = quantile(xx, q, na.rm=T)
    xmax = quantile(xx, 1 - q, na.rm=T)

    preds %>% ggplot(aes(x, predicted)) +
        geom_line() +
        geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
        stat_summary_bin(aes({{xvar}}, {{yvar}}), 
            data=data, fun.data=mean_se, bins=5, size=.2, 
            breaks=seq(xmin, xmax, length.out=bins),
        ) +
        facet_grid(~name) +
        labs(x=pretty_name(x), y=pretty_name(y)) +
        coord_cartesian(xlim=c(xmin, xmax))
}

regress_interaction = function(data, xvar, cvar, yvar, bins=6, bin_range=0.95) {
    # xstr = deparse(substitute(xvar)); cstr = deparse(substitute(cvar)); ystr = deparse(substitute(yvar))
    x = ensym(xvar); c = ensym(cvar); y = ensym(yvar)
    preds = data %>% 
        group_by(name) %>% 
        group_modify(function(data, grp) {
            model = if (grp$name == "Human") {
                model = inject(lmer(!!y ~ !!x * !!c + (!!x * !!c | wid), data=data))
                smart_print(summ(model))
                tibble(ggpredict(model, terms = c(glue("{x} [n=30]}"), as_string(c))))
            } else {
                model = inject(lm(!!y ~ !!x * !!c, data=data))
                tibble(ggpredict(model, terms = c(glue("{x} [n=30]}"), as_string(c))))
            }
        })

    xx = filter(data, name == "Human")[[x]]
    q = (1 - bin_range) / 2
    xmin = quantile(xx, q, na.rm=T)
    xmax = quantile(xx, 1 - q, na.rm=T)

    preds %>% ggplot(aes(x, predicted, group=group)) +
        geom_line(aes(color=group)) +
        geom_ribbon(aes(ymin=conf.low, ymax=conf.high), alpha=0.1) +
        stat_summary_bin(aes({{xvar}}, {{yvar}}, color={{cvar}}, group={{cvar}}), 
            data=data, fun.data=mean_se, bins=5, size=.2, 
            breaks=seq(xmin, xmax, length.out=bins),
        ) +
        facet_grid(~name) +
        theme(legend.position="top") +
        labs(x=pretty_name(x), y=pretty_name(y), color=pretty_name(c)) +
        coord_cartesian(xlim=c(xmin, xmax))
}


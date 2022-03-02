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
library(ggside)
library(broom.mixed)
library(lmerTest)

options(
    "summ-model.info"=FALSE, 
    "summ-model.fit"=FALSE, 
    "summ-re.table"=FALSE, 
    "summ-groups.table"=FALSE,
    "jtools-digits"=3,
    "max.print"=100
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

theme_set(theme_bw(base_size = 12))
theme_update(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    panel.grid.major.y = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(size=12),
    strip.text.y = element_text(size=12),
    legend.position="right",
    panel.spacing = unit(1, "lines"),
)
gridlines = theme(
    panel.grid.major.x = element_line(color="gray"),
    panel.grid.major.y = element_line(color="gray"),
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

ensure_column <- function(data, col) {
  add <-col[!col%in%names(data)]
  if(length(add)!=0) data[add] <- NA
  data
}

zscore = function(x) as.vector(scale(x))

midbins = function(x, breaks) {
    bin_ids = cut(x, breaks, labels=FALSE)
    left = breaks[-length(breaks)]
    right = breaks[-1]
    ((left + right) / 2)[bin_ids]
}

load_model_human = function(exp, name) {
    bind_rows(
        read_csv(glue('../data/processed/{exp}/{name}.csv'), 
            col_types = cols()) %>% mutate(name='Human'),
        read_csv(glue('../model/results/{exp}/optimal_{name}.csv'), 
            col_types = cols()) %>% mutate(name='Optimal'),
        read_csv(glue('../model/results/{exp}/empirical_{name}.csv'), 
            col_types = cols()) %>% mutate(name='Random'),
    ) %>% 
    mutate(name = factor(name, levels=c("Optimal", "Human", "Random"), ordered=T))
}

# %% ==================== Saving results ====================

sprintf_transformer <- function(text, envir) {
  m <- regexpr(":.+$", text)
  if (m != -1) {
    format <- substring(regmatches(text, m), 2)
    regmatches(text, m) <- ""
    res <- eval(parse(text = text, keep.source = FALSE), envir)
    do.call(sprintf, list(glue("%{format}f"), res))
  } else {
    eval(parse(text = text, keep.source = FALSE), envir)
  }
}

fmt <- function(..., .envir = parent.frame()) {
  glue(..., .transformer = sprintf_transformer, .envir = .envir)
}

pval = function(p) {
  # if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3)), 2)}")
  if (p < .001) "p < .001" else glue("p = {str_sub(format(round(p, 3), nsmall=3), 2)}")
}

tex_writer = function(path) {
  # dir.create(path, recursive=TRUE, showWarnings=FALSE)
  function(name, tex) {
    name = glue(name, .envir=parent.frame()) %>% str_replace("[:*]", "-")
    tex = fmt(tex, .envir=parent.frame())
    file = glue("{path}/{name}.tex")
    dir.create(dirname(file), recursive=TRUE, showWarnings=FALSE)
    print(paste0(file, ": ", tex))
    writeLines(paste0(tex, "\\unskip"), file)
  }
}

system('mkdir -p figs')
system('mkdir -p .fighist')

fig = function(name="tmp", w=4, h=4, dpi=320, pdf=FALSE, ...) {
    if (isTRUE(getOption('knitr.in.progress'))) {
        show(last_plot())
        return()
    }
    ggsave("/tmp/fig.png", width=w, height=h, dpi=dpi, ...)
    stamp = format(Sys.time(), "%m-%d-%H-%M-%S")
    p = glue('".fighist/{gsub("/", "-", name)}-{stamp}.png"')
    system(glue('mv /tmp/fig.png {p}'))
    system(glue('cp {p} figs/{name}.png'))
    if (pdf) ggsave(glue("figs/{name}.pdf"), width=w, height=h, ...)
    # invisible(dev.off())
    # knitr::include_graphics(p)
}

# %% ==================== Plotting ====================

plot_effect = function(df, x, y, color, min_n=10, geom="pointrange") {
    enough_data = df %>% 
        ungroup() %>% 
        filter(name == "Human") %>% 
        count({{color}}, {{x}}) %>% 
        filter(n > min_n)

    # enough_data = df %>% 
    #     ungroup() %>% 
    #     count(name, response_type, {{x}}) %>% 
    #     filter(n > min_n)
    warn("Using Normal approximation for confidence intervals", .frequency="once", .frequency_id="normconf")
    df %>% 
        right_join(enough_data) %>% 
        ggplot(aes({{x}}, {{y}}, color={{color}})) +
            stat_summary(fun=mean, geom="line") +
            stat_summary(fun.data=mean_cl_normal, geom=geom) +
            facet_wrap(~name)
            # theme(legend.position="none") +
            # pal +
}

# %% ==================== Tidy regressions ====================

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
tidyglmer = function(data, xvar, yvar, family="binomial") {
    y = ensym(yvar)
    x = ensym(xvar)
    inject(glmer(!!y ~ !!x + (!!x | wid), data=data, family=family))
}
tidymer = function(data, xvar, yvar) {
    binary_y = all(na.omit(data[[ensym(yvar)]]) %in% 0:1)
    fun = if (binary_y) tidyglmer else tidylmer
    fun(data, {{xvar}}, {{yvar}})
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
    rt_z = "Reaction Time (z-scored)",
    strength_z = "Strength (z-scored)",
    last_strength = "Last Fixated Strength"
)
pretty_labs = function(x, y) labs(x=pretty_name(x), y=pretty_name(y))

def_breaks = list(
    strength_first = seq(-3.5, 3.5),
    rel_strength = seq(-4.9, 4.9, 1.4)
)


pretty_name = function(x) {
    if (!is_null(pretty_names[[x]])) return(pretty_names[[x]])
    x %>% 
        str_replace_all("_", " ") %>% 
        str_replace_all("pres", "fixation") %>% 
        str_to_title %>% 
        gsub("(?!^)\\b(Of|In|The)\\b", "\\L\\1", ., perl=TRUE)
}

geom_xdensity = list(
    geom_xsidedensity(aes(y=stat(density))),
    scale_xsidey_continuous(breaks = NULL, labels = "")
)
geom_ydensity = list(
    geom_ysidedensity(aes(x=stat(density))),
    scale_ysidex_continuous(breaks = NULL, labels = "")
)

make_breaks = function(x, n=7, q=.025) {
    xmin = quantile(x, q, na.rm=T)
    xmax = quantile(x, 1 - q, na.rm=T)
    seq(xmin, xmax, length.out=n)
}

regress = function(data, xvar, yvar, bins=6, bin_range=0.95, mixed=TRUE, logistic=FALSE, show_reg=TRUE) {
    x = ensym(xvar); y = ensym(yvar)

    xx = filter(data, name == "Human")[[x]]
    q = (1 - bin_range) / 2
    xmin = quantile(xx, q, na.rm=T)
    xmax = quantile(xx, 1 - q, na.rm=T)

    preds = data %>% 
        group_by(name) %>% 
        group_modify(function(data, grp) {
            model = if (grp$name == "Human") {
                if (mixed) {
                    if (logistic) {
                        model = inject(glmer(!!y ~ !!x + (!!x | wid), family=binomial, data=data))
                    } else {
                        model = inject(lmer(!!y ~ !!x + (!!x | wid), data=data))
                    }
                } else {
                    if (logistic) {
                        model = inject(glm(!!y ~ !!x, family=binomial, data=data))
                    } else {
                        model = inject(lm(!!y ~ !!x, data=data))
                    }
                }
                if(show_reg) {
                    print(glue("N = {nrow(data)}"))
                    smart_print(summ(model))
                }
                tibble(ggpredict(model, terms = glue("{x} [n=100]}")))
            } else {
                if (logistic) {
                    model = inject(glm(!!y ~ !!x, family=binomial, data=data))
                } else {
                    model = inject(lm(!!y ~ !!x, data=data))
                }
                tibble(ggpredict(model, terms = glue("{x} [n=100]}")))
            }
        })

    pp = preds %>% filter(between(x, xmin, xmax)) 

    ggplot(data, aes({{xvar}}, {{yvar}})) +
        geom_line(aes(x, predicted), pp) +
        geom_ribbon(aes(x, predicted, ymin=conf.low, ymax=conf.high), pp, alpha=0.1) +
        stat_summary_bin(, 
            fun.data=mean_cl_normal, size=.2, 
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
            data=data, fun.data=mean_cl_normal, bins=5, size=.2, 
            breaks=seq(xmin, xmax, length.out=bins),
        ) +
        facet_grid(~name) +
        labs(x=pretty_name(x), y=pretty_name(y), color=pretty_name(c)) +
        coord_cartesian(xlim=c(xmin, xmax))
}


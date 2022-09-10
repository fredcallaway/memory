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
library(optigrab)

options(
    "summ-model.info"=FALSE, 
    "summ-model.fit"=FALSE, 
    "summ-re.table"=FALSE, 
    "summ-groups.table"=FALSE,
    "jtools-digits"=3,
    "max.print"=100
)

theme_set(theme_bw(base_size = 12))
theme_update(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
    # panel.grid.major.y = element_blank(),
    panel.grid.major.y = element_line(color="#EDEDED"),
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

# %% ==================== Miscellany ====================

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

n_pct = function(x) {
    glue("{sum(x)} ({round(100*mean(x))}\\%)")
}

quibble <- function(x, q = c(0.25, 0.5, 0.75)) {
  tibble(x = quantile(x, q), q = q)
}

midbins = function(x, breaks) {
    bin_ids = cut(x, breaks, labels=FALSE)
    left = breaks[-length(breaks)]
    right = breaks[-1]
    ((left + right) / 2)[bin_ids]
}

collapse_participants = function(data, f, y, ...) {
    data %>% 
        group_by(name, wid, ...) %>% 
        summarise("{{y}}" := f({{y}}, na.rm=T)) %>% 
        ungroup()
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
  function(tex, name) {
    name = glue(name, .envir=parent.frame()) %>% str_replace("[:*]", "-")
    tex = fmt(tex, .envir=parent.frame())
    file = glue("{path}/{name}.tex")
    dir.create(dirname(file), recursive=TRUE, showWarnings=FALSE)
    print(paste0(file, ": ", tex))
    writeLines(paste0(tex, "\\unskip"), file)
  }
}

system('mkdir -p .fighist')
fig = function(name="tmp", w=4, h=4, dpi=320, path="figs/", make_pdf=exists("MAKE_PDF") && MAKE_PDF, ...) {

    if (isTRUE(getOption('knitr.in.progress'))) {
        show(last_plot())
        return()
    }
    ggsave("/tmp/fig.png", width=w, height=h, dpi=dpi, ...)
    stamp = format(Sys.time(), "%m-%d-%H-%M-%S")
    p = glue('".fighist/{gsub("/", "-", name)}-{stamp}.png"')
    system(glue('mv /tmp/fig.png {p}'))
    system(glue('mkdir -p {path}'))
    system(glue('cp {p} {path}/{name}.png'))
    if (make_pdf) ggsave(glue("{path}/{name}.pdf"), width=w, height=h, ...)
    # invisible(dev.off())
    # knitr::include_graphics(p)
}

# %% ==================== Plotting ====================

mute = function(x, amt=.15) {
    x %>% 
        colorspace::lighten(amt) %>% 
        colorspace::desaturate(2*amt)
}

`-.gg` <- function(plot, layer) {
    if (missing(layer)) {
        stop("Cannot use `-.gg()` with a single argument. Did you accidentally put - on a new line?")
    }
    if (!is.ggplot(plot)) {
        stop('Need a plot on the left side')
    }
    plot$layers = c(layer, plot$layers)
    plot
}

plot_effect = function(df, x, y, color=NULL, collapser=mean, min_n=10, geom="pointrange") {
    dat = collapse_participants(df, collapser, {{y}}, {{x}}, {{color}})

    enough_data = dat %>% 
        ungroup() %>% 
        count(name, {{color}}, {{x}}) %>% 
        filter(name != "Human" | n > min_n)

    # warn("Using Normal approximation for confidence intervals", .frequency="once", .frequency_id="normconf")
    
    rng = dat %>% summarise(rng = max({{x}}) - min({{x}})) %>% with(rng[1])
    dodge = position_dodge2(.06 * rng)

    dat %>% 
        right_join(enough_data) %>% 
        ggplot(aes({{x}}, {{y}}, group={{color}}, color={{color}})) +
            stat_summary(fun=mean, geom="line", position=dodge) +
            stat_summary(fun.data=mean_cl_boot, geom=geom, position=dodge) +
            facet_wrap(~name)
            # theme(legend.position="none") +
            # pal +
}

plot_effect_continuous = function(data, x, y, color, collapser) {
    data %>% 
        ungroup() %>% 
        mutate(color = ordered({{color}})) %>% 
        collapse_participants(collapser, {{y}}, {{x}}, color) %>% 
        ggplot(aes({{x}}, {{y}}, group=color)) +
            stat_summary(aes(color=color), fun=mean, geom="line", size=.9) +
            stat_summary(fun.data=mean_cl_boot, geom="ribbon", alpha=0.08) +
            facet_wrap(~name) 
            # + 
            # theme(panel.grid.major.x = element_line(color="#EDEDED"))
}

join_limits = function(...) {
    # map(list(...), ~ layer_scales(.x)$y$range$range)
    ylo = list(...) %>% 
        map(~ layer_scales(.x)$y$range$range[[1]]) %>% 
        unlist %>% 
        min
    yhi = list(...) %>% 
        map(~ layer_scales(.x)$y$range$range[[2]]) %>% 
        unlist %>% 
        max
    c(ylo, yhi)
}

geom_xdensity = list(
    geom_xsidedensity(aes(y=stat(density))),
    scale_xsidey_continuous(breaks = NULL, labels = "")
)
geom_ydensity = list(
    geom_ysidedensity(aes(x=stat(density))),
    scale_ysidex_continuous(breaks = NULL, labels = "")
)

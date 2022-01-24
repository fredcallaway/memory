
regress = function(data, xvar, yvar, bins=6, bin_range=0.95, mixed=TRUE, logistic=FALSE) {
    x = ensym(xvar); y = ensym(yvar)

    xx = filter(data, name == "Human")[[x]]
    q = (1 - bin_range) / 2
    xmin = quantile(xx, q, na.rm=T)
    xmax = quantile(xx, 1 - q, na.rm=T)

    models = data %>% 
        group_by(pilot) %>% 
        group_modify(function(data, grp) {
            dd = transmute(data, x=zscore({{xvar}}), y={{yvar}}, wid)
            if (logistic) {
                model = glmer(y ~ x + (x | wid), family=binomial, data=dd)
                # model = glm(!!y ~ x, family=binomial, data=data)
            } else {
                model = lmer(y ~ x + (x | wid), data=dd)
                # model = lm(!!y ~ x, data=data)
            }
            summarise(data, μ=mean({{xvar}}), σ=sd({{xvar}}), model=list(model))
        })

    models %>% 
        rowwise() %>% 
        summarise(tidy(model)) %>% 
        filter(term != "(Intercept)" & effect == "fixed") %>% 
        select(pilot, estimate, std.error, p.value) %>% kable(digits=4) %>% smart_print

    preds = models %>% rowwise() %>% summarise(
        tibble(ggpredict(model, "x [n=100]")) %>% 
        mutate(x = σ * x + μ)  # un-standardize
    ) %>% filter(between(x, xmin, xmax))

    ggplot(data, aes({{xvar}}, {{yvar}})) +
        geom_line(aes(x, predicted), preds) +
        geom_ribbon(aes(x, predicted, ymin=conf.low, ymax=conf.high), preds, alpha=0.1) +
        stat_summary_bin(, 
            fun.data=mean_cl_normal, size=.2, 
            breaks=seq(xmin, xmax, length.out=bins),
        ) +
        facet_wrap(~pilot) +
        labs(x=pretty_name(x), y=pretty_name(y)) +
        coord_cartesian(xlim=c(xmin, xmax))
}

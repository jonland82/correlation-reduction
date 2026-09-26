# The Work of Reducing Dependence

[Read the paper](https://jonland82.github.io/correlation-reduction/) · [PDF](docs/paper.pdf) · [LaTeX source](source/paper.tex)

How much work does it take to reduce dependence between two systems? The paper follows one arc:

1. **Define the change.** The pointwise ratio $R=p_{12}/(p_1p_2)$ compares a joint distribution with independent marginals. Its average log is mutual information $I=\langle\log R\rangle$; $\Delta I=I_{\mathrm{before}}-I_{\mathrm{after}}$ is positive when dependence falls.
2. **Calculate work in equilibrium.** For two cross-coupled oscillators in one bath, slow isothermal control gives the exact, model-specific relation $W_{\mathrm{rev}}=k_BT\Delta I$.
3. **Model the membranes.** A cavity changes both coupling and local stiffness, while unequal baths drive heat flow. The covariance predicts that lowering coupling reduces *position* dependence but increases *full-state* dependence. Mechanical work follows the coupling ramp, $W_{\mathrm{on}}=\int\langle\partial H/\partial\Lambda\rangle\,d\Lambda$, so it cannot be read from $\Delta I$ alone.
4. **Test both together.** Published membrane measurements constrain the coupling and heat-flow scales. A direct test would record both membranes' joint motion and the calibrated ramp, then compare the measured dependence change and work for the same endpoints. That paired measurement remains proposed.

`docs/` contains the published HTML, PDF, and figures. `source/` contains the manuscript, calculations, data, and build scripts. `archive/` holds earlier drafts and exploratory results.

To rebuild the HTML, run `python source/generate_html_figures.py` and `python source/build_html.py` (requires NumPy, SciPy, Matplotlib, Beautiful Soup, and Pandoc). To rebuild the PDF, run `latexmk -cd -pdf source/paper.tex`, then copy `source/paper.pdf` to `docs/paper.pdf`.

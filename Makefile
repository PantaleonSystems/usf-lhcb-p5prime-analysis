.PHONY: all data fit consistency figures readme test clean distclean help

PYTHON ?= python
export PYTHONPATH := scripts

RAW := data/raw/HEPData-ins1409497-v1-yaml
CSV := data/p5p_observables.csv
FIT := results/fit_results.json
GLOBAL := results/global_consistency.json

help:
	@echo "make all          run the full pipeline from the raw HEPData YAML"
	@echo "make data         rebuild $(CSV)"
	@echo "make fit          run the MCMC fit and model comparison"
	@echo "make consistency  cross-check against Bs->mumu, R_K, R_K*"
	@echo "make figures      regenerate every figure"
	@echo "make readme       refresh the numbers quoted in README.md"
	@echo "make test         run the validation suite"
	@echo "make clean        remove generated results (keeps caches)"
	@echo "make distclean    also remove the flavio/response caches"

all: readme

$(CSV): scripts/generate_p5p_csv.py scripts/covariance.py $(RAW)/Table2.yaml
	$(PYTHON) scripts/generate_p5p_csv.py

data: $(CSV)

$(FIT): scripts/fit_usf.py scripts/sm_predictions.py scripts/response.py \
        scripts/utils.py $(CSV)
	$(PYTHON) scripts/fit_usf.py

fit: $(FIT)

$(GLOBAL): scripts/global_consistency.py $(FIT)
	$(PYTHON) scripts/global_consistency.py

consistency: $(GLOBAL)

figures: $(FIT)
	$(PYTHON) scripts/plot_p5p.py

readme: $(GLOBAL) figures
	$(PYTHON) scripts/update_readme.py

test:
	$(PYTHON) -m pytest -ra

clean:
	rm -f results/*.pdf results/*.png results/*.json results/*.h5

distclean: clean
	rm -f data/sm_cache.npz data/response_grid.npz

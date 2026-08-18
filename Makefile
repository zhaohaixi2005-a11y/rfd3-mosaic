.PHONY: format local-smoke local-test mosaic-public-check mosaic-build mosaic-release-smoke

# Use the currently activated environment on any workstation/server.  Fall
# back to the repository-local CPU environment when no venv is active.
LOCAL_PYTHON ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,.venv-local/bin/python)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled and cached files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .benchmarks

## Format src directory using black
format:
	ruff format src models tests
	ruff check --fix src models tests

## Verify the active local development environment and CLI.
local-smoke:
	@test -x "$(LOCAL_PYTHON)" || (echo "Python environment not found: $(LOCAL_PYTHON)"; false)
	@DEBUG=false TYPE_CHECK=false NAN_CHECK=true CCD_MIRROR_PATH= PDB_MIRROR_PATH= "$(LOCAL_PYTHON)" -c "import torch, rfd3, rfd3_mosaic; print('local environment: OK; torch=' + torch.__version__ + '; cuda=' + str(torch.cuda.is_available()))"
	@DEBUG=false TYPE_CHECK=false NAN_CHECK=true CCD_MIRROR_PATH= PDB_MIRROR_PATH= "$(LOCAL_PYTHON)" -m rfd3_mosaic.cli capabilities >/dev/null

## Run the complete RFD3-Mosaic CPU unit suite in the local environment.
local-test: local-smoke
	@DEBUG=false TYPE_CHECK=false NAN_CHECK=true CCD_MIRROR_PATH= PDB_MIRROR_PATH= "$(LOCAL_PYTHON)" -m unittest discover -s tests/rfd3_mosaic/unit -p 'test_*.py' -v

## Build the distributable RFD3-Mosaic wheel and source archive.
mosaic-build:
	@uv build

## Reject private deployment details and broken links in public documentation.
mosaic-public-check:
	@"$(LOCAL_PYTHON)" scripts/rfd3_mosaic/check_public_surface.py

## Verify CLI resources and imports in a wheel install outside the checkout.
mosaic-release-smoke: mosaic-public-check mosaic-build
	@bash scripts/rfd3_mosaic/release_smoke.sh

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

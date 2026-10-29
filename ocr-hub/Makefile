# OCR Hub — short commands over docker compose
.PHONY: up stop logs front clean build help

COMPOSE = docker compose --profile cpu

up: build ## build image + start → http://localhost:8990
	$(COMPOSE) up -d
	@echo "OCR Hub → http://localhost:8990"

build: ## (re)build the image
	$(COMPOSE) build

stop: ## stop the container (models kept in data/)
	$(COMPOSE) down

logs: ## follow backend logs
	$(COMPOSE) logs -f ocr-hub

front: ## rebuild frontend bundle and restart the container
	cd frontend && npm install --no-progress && npm run build
	rm -rf data/frontend_dist && mkdir -p data/frontend_dist
	cp -r frontend/dist/. data/frontend_dist/
	$(COMPOSE) restart ocr-hub

clean: ## remove container + downloaded models (fresh start)
	$(COMPOSE) down --rmi local
	rm -rf data/models/* data/results/* data/catalog.json
	@echo "Data reset. Settings (incl. HF token) kept in data/config.json."

help: ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

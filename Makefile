all:
	echo "Nothing to be done"

lint:
	ruff check .

format:
	ruff format .

test:
	bash query-test.sh --reset --verbose chat

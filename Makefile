.PHONY: docker dockerclean ci

docker:
	python setup.py sdist && \
	cp  `ls -rt dist/*.tar.gz|tail -1` docker/software-correlator.tar.gz && \
	docker build --force-rm=true --tag=cvqa:latest docker && \
	docker save -o cvqa.tar cvqa:latest &&\
	rm -f docker/software-correlator.tar.gz

dockerfull:
	python setup.py sdist && \
	cp  `ls -rt dist/*.tar.gz|tail -1` docker/software-correlator.tar.gz && \
	docker build --no-cache --force-rm=true --tag=cvqa:latest docker && \
	docker save -o cvqa.tar cvqa:latest &&\
	rm -f docker/software-correlator.tar.gz

dockerclean:
	docker rm -v $(docker ps -a -q -f status=exited);docker rmi  $(docker images -f "dangling=true" -q)


ci:
	while true; do python setup.py build; inotifywait softwarecorrelator/*.py setup.py MANIFEST.in Makefile;done

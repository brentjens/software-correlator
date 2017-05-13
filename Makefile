.PHONY: docker dockerclean

docker:
	python setup.py sdist && \
	cp  `ls -rt dist/*.tar.gz|tail -1` docker/software-correlator.tar.gz && \
	docker build --force-rm=true --shm-size=512MB --tag=cvqa:latest docker && \
	docker save -o cvqa.img cvqa:latest &&\
	rm -f docker/software-correlator.tar.gz

dockerclean:
	docker rm -v $(docker ps -a -q -f status=exited);docker rmi  $(docker images -f "dangling=true" -q)
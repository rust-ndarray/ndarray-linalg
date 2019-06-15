
registory    := termoshtt/tarpaulin

all: build

login:
	docker login

build:
	docker build . -t $(registory)

push: login
	docker push $(registory)

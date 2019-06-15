
rust_version := 1.35
registory    := termoshtt/tarpaulin

all: build

login:
	docker login

build:
	docker build . -t $(registory):$(rust_version)

push: login
	docker push $(registory):$(rust_version)

#!/bin/bash

rm -rf target/cov
for t in target/debug/*[^\.d]; do
  echo "kcov.bash: ${t}"
  cov_out="target/cov/$(basename $t)"
  mkdir -p "$cov_out"
  kcov --exclude-pattern=/.cargo,/usr/lib --verify "$cov_out" "$t"
done

bash <(curl -s https://codecov.io/bash) && echo "Uploaded code coverage"

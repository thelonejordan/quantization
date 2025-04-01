# Quantization

clone [tinygrad](https://github.com/tinygrad/tinygrad) inside `./build` directory

```shell
mkdir build && cd build
git clone git@github.com:tinygrad/tinygrad.git
cd -
```

install dependencies

```shell
pip install -r requirements.txt
```

run

```shell
COUNT=10 python basic.py
```

```shell
COUNT=10 ERROR=1 python basic.py
```

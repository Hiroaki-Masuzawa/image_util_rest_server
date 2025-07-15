# 
REST APIでモデルでインスタンスセグメンテーションを行う．


## server 
### docker build
```
cd server 
./build.sh
```
### run server 
```
cd server 
./run_server.sh
```
### kill server 
```
cd server 
./kill_server.sh
```

## client
```
cd client
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
python3 client_test.py --image input.jpg --server localhost --port 8008 --model maskrcnn --debug
python3 client_test.py --image input.jpg --server localhost --port 8008 --model ram-grounded-sam --debug
python3 client_test.py --image input.jpg --server localhost --port 8008 --model ram-grounded-sam --prompt "horse,person" --debug
```
`--debug`を付けると`output.png`に結果が示される
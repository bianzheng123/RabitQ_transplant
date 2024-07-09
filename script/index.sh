
C=4096
D=128
B=128
username='bianzheng'
dataset='siftsmall'

g++ -o ./bin/index_${dataset} ./src/index.cpp -I ./src/ -O3 -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN

./bin/index_${dataset} -d $dataset -u $username
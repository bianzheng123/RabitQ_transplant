

username='bianzheng'
dataset='siftsmall'
C=4096
B=128
D=128
k=100

g++ -march=core-avx2 -Ofast -o ./bin/search_${dataset} ./src/search.cpp -I ./src/ -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D FAST_SCAN

result_path=/home/${username}/RaBitQ/results
mkdir ${result_path}

res="${result_path}/${dataset}/"

mkdir "$result_path/${dataset}/"

./bin/search_${dataset} -d ${dataset} -r ${res} -k ${k} -u ${username} -o yes

./bin/search_${dataset} -d ${dataset} -r ${res} -k ${k} -u ${username} -o no

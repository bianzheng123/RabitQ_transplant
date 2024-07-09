import os

if __name__ == '__main__':
    dataset_info_m = {
        'siftsmall': {
            'D': 128
        },
        'sift': {
            'D': 128
        },
        'deep': {
            'D': 256
        },
        'glove': {
            'D': 200
        }
    }

    username = 'bianzheng'
    # dataset_l = ['siftsmall', 'sift', 'deep', 'glove']
    dataset_l = ['siftsmall']
    for i in range(1):
        for dataset in dataset_l:
            C = 4096
            D = dataset_info_m[dataset]['D']
            B = (D + 63) // 64 * 64
            k = 100

            # os.system(
            #     f'g++ -march=core-avx2 -Ofast -o ./bin/search_{dataset} ./src/search.cpp -I ./src/ -D BB={B} -D DIM={D} -D numC={C} -D B_QUERY=4 -D FAST_SCAN')
            os.system(
                f'cd /home/{username}/RaBitQ/build && cmake .. && make')
            result_path = f'/home/{username}/RaBitQ/results'
            os.makedirs(result_path, exist_ok=True)

            os.makedirs(f"{result_path}/{dataset}/", exist_ok=True)

            result_dataset_path = f"{result_path}/{dataset}/"

            # os.system(f'./bin/search_{dataset} -d {dataset} -r {result_dataset_path} -k {k} -u {username} -o yes')
            # os.system(f'./bin/search_{dataset} -d {dataset} -r {result_dataset_path} -k {k} -u {username} -o no')

            os.system(f'./build/search_{dataset} -d {dataset} -r {result_dataset_path} -k {k} -u {username} -o yes')
            os.system(f'./build/search_{dataset} -d {dataset} -r {result_dataset_path} -k {k} -u {username} -o no')

            command = """
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
            """

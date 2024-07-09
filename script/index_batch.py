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
    for dataset in dataset_l:
        C = 4096
        D = dataset_info_m[dataset]['D']
        B = (D + 63) // 64 * 64

        os.system(f"cd /home/{username}/RaBitQ/build && cmake .. && make")
        os.system(f'./build/index_{dataset} -d {dataset} -u {username}')

        # os.system(
        #     f'g++ -o ./bin/index_{dataset} ./src/index.cpp -I ./src/ -O3 -march=core-avx2 -D BB={B} -D DIM={D} -D numC={C} -D B_QUERY=4 -D SCAN')
        # os.system(f'./bin/index_{dataset} -d {dataset} -u {username}')

        command = """
        C=4096
        D=128
        B=128
        username='bianzheng'
        dataset='siftsmall'
        
        g++ -o ./bin/index_${dataset} ./src/index.cpp -I ./src/ -O3 -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN
        
        ./bin/index_${dataset} -d $dataset -u $username
        """

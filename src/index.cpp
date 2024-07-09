#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2

#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"

using namespace std;

int main(int argc, char *argv[]) {

    const struct option longopts[] = {
            // General Parameter
            {"help",     no_argument,       0, 'h'},

            // Indexing Path
            {"dataset",  required_argument, 0, 'd'},
            {"username", required_argument, 0, 'u'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char dataset[256] = "";
    char username[256] = "";

    while (iarg != -1) {
        iarg = getopt_long(argc, argv, "d:u:", longopts, &ind);
        switch (iarg) {
            case 'd':
                if (optarg) {
                    strcpy(dataset, optarg);
                }
                break;
            case 'u':
                if (optarg) {
                    strcpy(username, optarg);
                }
                break;
        }
    }
    char data_path[256] = "";
    char index_path[256] = "";
    sprintf(data_path, "/home/%s/RaBitQ/data/%s", username, dataset);
    sprintf(index_path, "/home/%s/RaBitQ/index/%s", username, dataset);



    // ==============================================================================================================
    // Load Data
    char base_data_filename[256] = "";
    char result_index_filename[256] = "";
    char centroid_path[256] = "";
    char x0_path[256] = "";
    char dist_to_centroid_path[256] = "";
    char cluster_id_path[256] = "";
    char binary_path[256] = "";

    sprintf(base_data_filename, "%s/%s_base.fvecs", data_path, dataset);
    Matrix<float> X(base_data_filename);

    sprintf(centroid_path, "%s/RandCentroid_C%d_B%d.fvecs", index_path, numC, BB);
    Matrix<float> C(centroid_path);

    sprintf(x0_path, "%s/x0_C%d_B%d.fvecs", index_path, numC, BB);
    Matrix<float> x0(x0_path);

    sprintf(dist_to_centroid_path, "%s/%s_dist_to_centroid_%d.fvecs", index_path, dataset, numC);
    Matrix<float> dist_to_centroid(dist_to_centroid_path);

    sprintf(cluster_id_path, "%s/%s_cluster_id_%d.ivecs", index_path, dataset, numC);
    Matrix<uint32_t> cluster_id(cluster_id_path);

    sprintf(binary_path, "%s/RandNet_C%d_B%d.Ivecs", index_path, numC, BB);
    Matrix<uint64_t> binary(binary_path);

    sprintf(result_index_filename, "%s/ivfrabitq%d_B%d.index", index_path, numC, BB);
    std::cerr << "Loading Succeed!" << std::endl << std::endl;
    // ==============================================================================================================

    IVFRN<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);

    ivf.save(result_index_filename);

    // ==============================================================================================================
    // No rotation

    sprintf(centroid_path, "%s/no_rotation/RandCentroid_C%d_B%d.fvecs", index_path, numC, BB);
    Matrix<float> C_n(centroid_path);

    sprintf(x0_path, "%s/no_rotation/x0_C%d_B%d.fvecs", index_path, numC, BB);
    Matrix<float> x0_n(x0_path);

    sprintf(binary_path, "%s/no_rotation/RandNet_C%d_B%d.Ivecs", index_path, numC, BB);
    Matrix<uint64_t> binary_n(binary_path);

    sprintf(result_index_filename, "%s/no_rotation/ivfrabitq%d_B%d.index", index_path, numC, BB);
    std::cerr << "Loading Succeed!" << std::endl << std::endl;
    // ==============================================================================================================

    IVFRN<DIM, BB> ivf_no_rotation(X, C_n, dist_to_centroid, x0_n, cluster_id, binary_n);

    ivf_no_rotation.save(result_index_filename);

    return 0;
}

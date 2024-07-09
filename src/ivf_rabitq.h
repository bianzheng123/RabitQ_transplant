// ==================================================================
// IVFRN() involves pre-processing steps (e.g., packing the
// quantization codes into a batch) in the index phase.

// search() is the main function of the query phase.
// ==================================================================
#pragma once
#define RANDOM_QUERY_QUANTIZATION

#include <queue>
#include <vector>
#include <algorithm>
#include <map>
#include "matrix.h"
#include "utils.h"
#include "space.h"
#include "fast_scan.h"

class IVFRN {
private:
public:
    struct Factor {
        float sqr_x;
        float error;
        float factor_ppc;
        float factor_ip;
    };

    Factor *fac;
    float fac_norm;
    float max_x1;

    Space space;

    int vec_dim_, vec_dim_pad_;

    uint32_t N;                       // the number of data vectors 
    uint32_t C;                       // the number of clusters

    uint32_t *start;                  // the start point of a cluster
    uint32_t *packed_start;           // the start point of a cluster (packed with batch of 32)
    uint32_t *len;                    // the length of a cluster
    uint32_t *id;                     // N of size_t the ids of the objects in a cluster
    float *dist_to_c;                // N of floats distance to the centroids (not the squared distance)
    float *u;                        // B of floats random numbers sampled from the uniform distribution [0,1]

    uint64_t *binary_code;           // (B / 64) * N of 64-bit uint64_t
    uint8_t *packed_code;            // packed code with the batch size of 32 vectors


    float *x0;                       // N of floats in the Random Net algorithm
    float *centroid;                 // N * B floats (not N * D), note that the centroids should be randomized
    float *data;                     // N * D floats, note that the datas are not randomized

    IVFRN();

    IVFRN(const int vec_dim, const int vec_dim_pad);

    IVFRN(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &dist_to_centroid,
          const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary,
          const int vec_dim, const int vec_dim_pad);

    ~IVFRN();

    ResultHeap search(float *query, float *rd_query, uint32_t k, uint32_t nprobe,
                      float distK = std::numeric_limits<float>::max()) const;

    void scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint64_t *quant_query, uint64_t *ptr_binary_code, uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id);

    void fast_scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code, uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id) const;

    void save(char *filename);

    void load(char *filename);
};

// scan impl
void IVFRN::scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint64_t *quant_query, uint64_t *ptr_binary_code, uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id) {

    constexpr int SIZE = 32;
    float y = std::sqrt(sqr_y);
    float res[SIZE];
    float *ptr_res = &res[0];
    int it = len / SIZE;

    for (int i = 0; i < it; i++) {
        ptr_res = &res[0];
        for (int j = 0; j < SIZE; j++) {
            float tmp_dist = (ptr_fac->sqr_x) + sqr_y + ptr_fac->factor_ppc * vl +
                             (space.ip_byte_bin(quant_query, ptr_binary_code) * 2 - sumq) * (ptr_fac->factor_ip) *
                             width;
            float error_bound = y * (ptr_fac->error);
            *ptr_res = tmp_dist - error_bound;
            ptr_binary_code += vec_dim_pad_ / 64;
            ptr_fac++;
            ptr_res++;
        }

        ptr_res = &res[0];
        for (int j = 0; j < SIZE; j++) {
            if (*ptr_res < distK) {

                float gt_dist = sqr_dist(query, data, vec_dim_);
                if (gt_dist < distK) {
                    KNNs.emplace(gt_dist, *id);
                    if (KNNs.size() > k) KNNs.pop();
                    if (KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += vec_dim_;
            ptr_res++;
            id++;
        }
    }

    ptr_res = &res[0];
    for (int i = it * SIZE; i < len; i++) {
        float tmp_dist = (ptr_fac->sqr_x) + sqr_y + ptr_fac->factor_ppc * vl +
                         (space.ip_byte_bin(quant_query, ptr_binary_code) * 2 - sumq) * (ptr_fac->factor_ip) * width;
        float error_bound = y * (ptr_fac->error);
        *ptr_res = tmp_dist - error_bound;
        ptr_binary_code += vec_dim_pad_ / 64;
        ptr_fac++;
        ptr_res++;
    }

    ptr_res = &res[0];
    for (int i = it * SIZE; i < len; i++) {
        if (*ptr_res < distK) {
            float gt_dist = sqr_dist(query, data, vec_dim_);
            if (gt_dist < distK) {
                KNNs.emplace(gt_dist, *id);
                if (KNNs.size() > k) KNNs.pop();
                if (KNNs.size() == k)distK = KNNs.top().first;
            }
        }
        data += vec_dim_;
        ptr_res++;
        id++;
    }
}

void IVFRN::fast_scan(ResultHeap &KNNs, float &distK, uint32_t k, \
                        uint8_t *LUT, uint8_t *packed_code, uint32_t len, Factor *ptr_fac, \
                        const float sqr_y, const float vl, const float width, const float sumq, \
                        float *query, float *data, uint32_t *id) const {

    for (int i = 0; i < vec_dim_pad_ / 4 * 16; i++)LUT[i] *= 2;

    float y = std::sqrt(sqr_y);

    constexpr uint32_t SIZE = 32;
    uint32_t it = len / SIZE;
    uint32_t remain = len - it * SIZE;
    uint32_t nblk_remain = (remain + 31) / 32;

    while (it--) {
        float low_dist[SIZE];
        float *ptr_low_dist = &low_dist[0];
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate((SIZE / 32), packed_code, LUT, result, vec_dim_pad_);
        packed_code += SIZE * vec_dim_pad_ / 8;

        for (int i = 0; i < SIZE; i++) {
            float tmp_dist = (ptr_fac->sqr_x) + sqr_y + ptr_fac->factor_ppc * vl +
                             (result[i] - sumq) * (ptr_fac->factor_ip) * width;
            float error_bound = y * (ptr_fac->error);
            *ptr_low_dist = tmp_dist - error_bound;
            ptr_fac++;
            ptr_low_dist++;
        }
        ptr_low_dist = &low_dist[0];
        for (int j = 0; j < SIZE; j++) {
            if (*ptr_low_dist < distK) {

                float gt_dist = sqr_dist(query, data, vec_dim_);
                // cerr << *ptr_low_dist << " " << gt_dist << endl;
                if (gt_dist < distK) {
                    KNNs.emplace(gt_dist, *id);
                    if (KNNs.size() > k) KNNs.pop();
                    if (KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += vec_dim_;
            ptr_low_dist++;
            id++;
        }
    }

    {
        float low_dist[SIZE];
        float *ptr_low_dist = &low_dist[0];
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate(nblk_remain, packed_code, LUT, result, vec_dim_pad_);

        for (int i = 0; i < remain; i++) {
            float tmp_dist = (ptr_fac->sqr_x) + sqr_y + ptr_fac->factor_ppc * vl +
                             (result[i] - sumq) * ptr_fac->factor_ip * width;
            float error_bound = y * (ptr_fac->error);

            // ***********************************************************************************************
            *ptr_low_dist = tmp_dist - error_bound;
            ptr_fac++;
            ptr_low_dist++;
        }
        ptr_low_dist = &low_dist[0];
        for (int i = 0; i < remain; i++) {
            if (*ptr_low_dist < distK) {

                float gt_dist = sqr_dist(query, data, vec_dim_);
                if (gt_dist < distK) {
                    KNNs.emplace(gt_dist, *id);
                    if (KNNs.size() > k) KNNs.pop();
                    if (KNNs.size() == k)distK = KNNs.top().first;
                }
            }
            data += vec_dim_;
            ptr_low_dist++;
            id++;
        }
    }
}

// search impl
ResultHeap IVFRN::search(float *query, float *rd_query, uint32_t k, uint32_t nprobe, float distK) const {
    // The default value of distK is +inf 
    ResultHeap KNNs;
    // ===========================================================================================================
    // Find out the nearest N_{probe} centroids to the query vector.
    Result centroid_dist[numC];
    float *ptr_c = centroid;
    for (int i = 0; i < C; i++) {
        centroid_dist[i].first = sqr_dist(rd_query, ptr_c, vec_dim_pad_);
        centroid_dist[i].second = i;
        ptr_c += vec_dim_pad_;
    }
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + numC);

    // ===========================================================================================================
    // Scan the first nprobe clusters.
    Result *ptr_centroid_dist = (&centroid_dist[0]);
    uint8_t  PORTABLE_ALIGN64 byte_query[vec_dim_pad_];

    for (int pb = 0; pb < nprobe; pb++) {
        uint32_t c = ptr_centroid_dist->second;
        float sqr_y = ptr_centroid_dist->first;
        ptr_centroid_dist++;

        // =======================================================================================================
        // Preprocess the residual query and the quantized query
        float vl, vr;
        space.range(rd_query, centroid + c * vec_dim_pad_, vl, vr);
        float width = (vr - vl) / ((1 << B_QUERY) - 1);
        uint32_t sum_q = 0;
        space.quantize(byte_query, rd_query, centroid + c * vec_dim_pad_, u, vl, width, sum_q);

#if defined(SCAN)               // Binary String Representation 
        uint64_t PORTABLE_ALIGN32 quant_query[B_QUERY * vec_dim_pad_ / 64];
        memset(quant_query, 0, sizeof(quant_query));
        space.transpose_bin(byte_query, quant_query);
#elif defined(FAST_SCAN)        // Look-Up-Table Representation 
        uint8_t PORTABLE_ALIGN32 LUT[vec_dim_pad_ / 4 * 16];
        pack_LUT(byte_query, LUT,
                 vec_dim_pad_);
#endif

#if defined(SCAN)
        scan(KNNs, distK, k,\
                quant_query, binary_code + start[c] * (vec_dim_pad_ / 64), len[c], fac + start[c], \
                sqr_y, vl, width, sum_q,\
                query, data + start[c] * vec_dim_, id + start[c]);
#elif defined(FAST_SCAN)
        fast_scan(KNNs, distK, k, \
                LUT, packed_code + packed_start[c], len[c], fac + start[c], \
                sqr_y, vl, width, sum_q, \
                query, data + start[c] * vec_dim_, id + start[c]);
#endif
    }
    return KNNs;
}


// ==============================================================================================================================
// Save and Load Functions
void IVFRN::save(char *filename) {
    std::ofstream output(filename, std::ios::binary);

    uint32_t d = vec_dim_;
    uint32_t b = vec_dim_pad_;
    output.write((char *) &N, sizeof(uint32_t));
    output.write((char *) &d, sizeof(uint32_t));
    output.write((char *) &C, sizeof(uint32_t));
    output.write((char *) &b, sizeof(uint32_t));

    output.write((char *) start, C * sizeof(uint32_t));
    output.write((char *) len, C * sizeof(uint32_t));
    output.write((char *) id, N * sizeof(uint32_t));
    output.write((char *) dist_to_c, N * sizeof(float));
    output.write((char *) x0, N * sizeof(float));

    output.write((char *) centroid, C * vec_dim_pad_ * sizeof(float));
    output.write((char *) data, N * vec_dim_ * sizeof(float));
    output.write((char *) binary_code, N * vec_dim_pad_ / 64 * sizeof(uint64_t));

    output.close();
    std::cerr << "Saved!" << std::endl;
}

// load impl
void IVFRN::load(char *filename) {
    std::ifstream input(filename, std::ios::binary);
    //std::cerr << filename << std::endl;

    if (!input.is_open())
        throw std::runtime_error("Cannot open file");

    uint32_t d;
    uint32_t b;
    input.read((char *) &N, sizeof(uint32_t));
    input.read((char *) &d, sizeof(uint32_t));
    input.read((char *) &C, sizeof(uint32_t));
    input.read((char *) &b, sizeof(uint32_t));

    std::cerr << d << std::endl;
    assert(d == vec_dim_);
    assert(b == vec_dim_pad_);

    u = new float[vec_dim_pad_];
#if defined(RANDOM_QUERY_QUANTIZATION)
//    std::random_device rd;
//    std::mt19937 gen(rd());
    std::mt19937 gen(0);
    std::uniform_real_distribution<> uniform(0.0, 1.0);
    for (int i = 0; i < vec_dim_pad_; i++)u[i] = uniform(gen);
#else
    for(int i=0;i<vec_dim_pad_;i++)u[i] = 0.5;
#endif

    centroid = new float[C * vec_dim_pad_];
    data = new float[N * vec_dim_];

    binary_code = static_cast<uint64_t *>(aligned_alloc(256, N * vec_dim_pad_ / 64 * sizeof(uint64_t)));

    start = new uint32_t[C];
    len = new uint32_t[C];
    id = new uint32_t[N];
    dist_to_c = new float[N];
    x0 = new float[N];

    fac = new Factor[N];

    input.read((char *) start, C * sizeof(uint32_t));
    input.read((char *) len, C * sizeof(uint32_t));
    input.read((char *) id, N * sizeof(uint32_t));
    input.read((char *) dist_to_c, N * sizeof(float));
    input.read((char *) x0, N * sizeof(float));

    input.read((char *) centroid, C * vec_dim_pad_ * sizeof(float));
    input.read((char *) data, N * vec_dim_ * sizeof(float));
    input.read((char *) binary_code, N * vec_dim_pad_ / 64 * sizeof(uint64_t));

#if defined(FAST_SCAN)
    packed_start = new uint32_t[C];
    int cur = 0;
    for (int i = 0; i < C; i++) {
        packed_start[i] = cur;
        cur += (len[i] + 31) / 32 * 32 * vec_dim_pad_ / 8;
    }
    packed_code = static_cast<uint8_t *>(aligned_alloc(32, cur * sizeof(uint8_t)));
    for (int i = 0; i < C; i++) {
        pack_codes(binary_code + start[i] * (vec_dim_pad_ / 64), len[i], packed_code + packed_start[i],
                   vec_dim_pad_);
    }
#else
    packed_start = NULL;
    packed_code  = NULL;
#endif
    for (int i = 0; i < N; i++) {
        long double x_x0 = (long double) dist_to_c[i] / x0[i];
        fac[i].sqr_x = dist_to_c[i] * dist_to_c[i];
        fac[i].error = 2 * max_x1 * std::sqrt(x_x0 * x_x0 - dist_to_c[i] * dist_to_c[i]);
        fac[i].factor_ppc =
                -2 / fac_norm * x_x0 * ((float) space.popcount(binary_code + i * vec_dim_pad_ / 64) * 2 - vec_dim_pad_);
        fac[i].factor_ip = -2 / fac_norm * x_x0;
    }
    input.close();
}


// ==============================================================================================================================
// Construction and Deconstruction Functions
IVFRN::IVFRN() {

    IVFRN::vec_dim_ = 128;
    IVFRN::vec_dim_pad_ = 128;
    fac_norm = const_sqrt(1.0 * vec_dim_pad_);
    max_x1 = 1.9 / const_sqrt(1.0 * vec_dim_pad_ - 1.0);

    space = Space(vec_dim_, vec_dim_pad_);

    N = C = 0;
    start = len = id = NULL;
    x0 = dist_to_c = centroid = data = NULL;
    binary_code = NULL;
    fac = NULL;
    u = NULL;
}

IVFRN::IVFRN(const int vec_dim, const int vec_dim_pad) {

    IVFRN::vec_dim_ = vec_dim;
    IVFRN::vec_dim_pad_ = vec_dim_pad;
    fac_norm = const_sqrt(1.0 * vec_dim_pad);
    max_x1 = 1.9 / const_sqrt(1.0 * vec_dim_pad - 1.0);

    space = Space(vec_dim, vec_dim_pad);

    N = C = 0;
    start = len = id = NULL;
    x0 = dist_to_c = centroid = data = NULL;
    binary_code = NULL;
    fac = NULL;
    u = NULL;
}

IVFRN::IVFRN(const Matrix<float> &X, const Matrix<float> &_centroids, const Matrix<float> &dist_to_centroid,
             const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary,
             const int vec_dim, const int vec_dim_pad) {

    IVFRN::vec_dim_ = vec_dim;
    IVFRN::vec_dim_pad_ = vec_dim_pad;
    fac_norm = const_sqrt(1.0 * vec_dim_pad);
    max_x1 = 1.9 / const_sqrt(1.0 * vec_dim_pad - 1.0);

    space = Space(vec_dim, vec_dim_pad);

    fac = NULL;
    u = NULL;

    N = X.n;
    C = _centroids.n;

    // check uint64_t
    assert(vec_dim_pad_ % 64 == 0);
    assert(vec_dim_pad_ >= vec_dim_);

    start = new uint32_t[C];
    len = new uint32_t[C];
    id = new uint32_t[N];
    dist_to_c = new float[N];
    x0 = new float[N];

    memset(len, 0, C * sizeof(uint32_t));
    for (int i = 0; i < N; i++)len[cluster_id.data[i]]++;
    int sum = 0;
    for (int i = 0; i < C; i++) {
        start[i] = sum;
        sum += len[i];
    }
    for (int i = 0; i < N; i++) {
        id[start[cluster_id.data[i]]] = i;
        dist_to_c[start[cluster_id.data[i]]] = dist_to_centroid.data[i];
        x0[start[cluster_id.data[i]]] = _x0.data[i];
        start[cluster_id.data[i]]++;
    }
    for (int i = 0; i < C; i++) {
        start[i] -= len[i];
    }

    centroid = new float[C * vec_dim_pad_];
    data = new float[N * vec_dim_];
    binary_code = new uint64_t[N * vec_dim_pad_ / 64];

    std::memcpy(centroid, _centroids.data, C * vec_dim_pad_ * sizeof(float));
    float *data_ptr = data;
    uint64_t *binary_code_ptr = binary_code;

    for (int i = 0; i < N; i++) {
        int x = id[i];
        std::memcpy(data_ptr, X.data + x * vec_dim_, vec_dim_ * sizeof(float));
        std::memcpy(binary_code_ptr, binary.data + x * (vec_dim_pad_ / 64), (vec_dim_pad_ / 64) * sizeof(uint64_t));
        data_ptr += vec_dim_;
        binary_code_ptr += vec_dim_pad_ / 64;
    }
}

IVFRN::~IVFRN() {
    if (id != NULL) delete[] id;
    if (dist_to_c != NULL) delete[] dist_to_c;
    if (len != NULL) delete[] len;
    if (start != NULL) delete[] start;
    if (x0 != NULL) delete[] x0;
    if (data != NULL) delete[] data;
    if (fac != NULL) delete[] fac;
    if (u != NULL) delete[] u;
    if (binary_code != NULL) std::free(binary_code);
    // if(pack_codes  != NULL) std::free(pack_codes);
    if (centroid != NULL) std::free(centroid);
}

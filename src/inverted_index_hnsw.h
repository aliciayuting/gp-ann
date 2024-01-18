#pragma once

#include "defs.h"
#include "topn.h"

#include "../external/hnswlib/hnswlib/hnswlib.h"

#include <parlay/parallel.h>

struct InvertedIndexHNSW {
#ifdef MIPS_DISTANCE
    hnswlib::InnerProductSpace space;
#else
    hnswlib::L2Space space;
#endif

    std::vector<hnswlib::HierarchicalNSW<float> *> bucket_hnsws;


    HNSWParameters hnsw_parameters;

    InvertedIndexHNSW(PointSet& points, const Clusters& clusters) : space(points.d) {
        size_t num_shards = clusters.size();
        bucket_hnsws.resize(num_shards);
        size_t total_insertions = 0;
        for (int b = 0; b < num_shards; ++b) {
            bucket_hnsws[b] = new hnswlib::HierarchicalNSW<float>(
                &space, clusters[b].size(),
                hnsw_parameters.M, hnsw_parameters.ef_construction,
                /* random_seed = */ 555 + b);

            bucket_hnsws[b]->setEf(hnsw_parameters.ef_search);
            total_insertions += clusters[b].size();
        }

        std::cout << "start HNSW insertions" << std::endl;

        size_t x = 0;
        parlay::parallel_for(0, clusters.size(), [&](size_t b) {
            parlay::parallel_for(0, clusters[b].size(), [&](size_t i_local) {
                float* p = points.GetPoint(clusters[b][i_local]);
                bucket_hnsws[b]->addPoint(p, i_local);
                size_t x1 = __atomic_fetch_add(&x, 1, __ATOMIC_RELAXED);
                if (x1 % 50000 == 0) {
                    std::cout << "finished " << x1 << " / " << total_insertions << " HNSW insertions" << std::endl;
                }
            });
        });
    }

    ~InvertedIndexHNSW() { for (size_t i = 0; i < bucket_hnsws.size(); ++i) { delete bucket_hnsws[i]; } }

    NNVec Query(float* Q, int num_neighbors, const std::vector<int>& buckets_to_probe, int num_probes) const {
        TopN top_k(num_neighbors);
        for (int i = 0; i < num_probes; ++i) {
            const int bucket = buckets_to_probe[i];
            auto result = bucket_hnsws[bucket]->searchKnn(Q, num_neighbors);
            while (!result.empty()) {
                const auto [dist, label] = result.top();
                result.pop();
                top_k.Add(std::make_pair(dist, label));
            }
        }
        return top_k.Take();
    }
};

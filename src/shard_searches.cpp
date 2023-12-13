#include "shard_searches.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include "defs.h"
#include "../external/hnswlib/hnswlib/hnswlib.h"
#include <parlay/parallel.h>
#include <parlay/sequence.h>

std::vector<ShardSearch> RunInShardSearches(PointSet& points, PointSet& queries, HNSWParameters hnsw_parameters, int num_neighbors,
                                            const Clusters& clusters, int num_shards, const std::vector<float>& distance_to_kth_neighbor) {
    std::vector<size_t> ef_search_param_values = { 50, 80, 100, 150, 200, 250, 300, 400, 500 };

    Timer init_timer;
    init_timer.Start();
    std::vector<ShardSearch> shard_searches(ef_search_param_values.size());
    for (size_t i = 0; i < ef_search_param_values.size(); ++i) { shard_searches[i].Init(ef_search_param_values[i], num_shards, queries.n); }
    std::cout << "Init search output took " << init_timer.Stop() << std::endl;

    for (int b = 0; b < num_shards; ++b) {
        auto cluster = clusters[b];

        std::cout << "Start building HNSW for shard " << b << " of size " << cluster.size() << std::endl;

#ifdef MIPS_DISTANCE
        using SpaceType = hnswlib::InnerProductSpace;
#else
        using SpaceType = hnswlib::L2Space;
#endif

        SpaceType space(points.d);

        Timer build_timer;
        build_timer.Start();
        hnswlib::HierarchicalNSW<float> hnsw(&space, cluster.size(), hnsw_parameters.M, hnsw_parameters.ef_construction, 555 + b);

        std::mt19937 prng(555 + b);
        std::shuffle(cluster.begin(), cluster.end(), prng);

        // do some insertion sequentially
        const size_t seq_insertion = std::min(1UL << 11, cluster.size());
        for (size_t i = 0; i < seq_insertion; ++i) { hnsw.addPoint(points.GetPoint(cluster[i]), i); }
        parlay::parallel_for(seq_insertion, cluster.size(), [&](size_t i) { hnsw.addPoint(points.GetPoint(cluster[i]), i); }, 512);

        std::cout << "HNSW build took " << build_timer.Stop() << std::endl;

        parlay::execute_with_scheduler(std::min<size_t>(32, parlay::num_workers()), [&] {
            size_t ef_search_param_id = 0;
            for (const size_t ef_search : ef_search_param_values) {
                parlay::sequence<std::priority_queue<std::pair<float, unsigned long>>> results(queries.n);

                hnsw.setEf(ef_search);
                Timer total;
                total.Start();
                parlay::parallel_for(0, queries.n, [&](size_t q) {
                    results[q] = hnsw.searchKnn(queries.GetPoint(q), num_neighbors);
                }, 10);
                const double elapsed = total.Stop();

                size_t total_hits = 0;
                parlay::parallel_for(0, queries.n, [&](size_t q) {
                    // a not so nice hack, but there is no other way to measure parallel runtime, if we don't
                    // want to repeat the query for each probe config (which we don't because it would take forever.
                    // this is the parameter tuning code after all.)
                    shard_searches[ef_search_param_id].time_query_in_shard[b][q] = elapsed / queries.n;

                    // now transfer the neighbors
                    auto& nn = shard_searches[ef_search_param_id].neighbors[b][q];
                    auto& pq = results[q];
                    size_t hits = 0;
                    while (!pq.empty()) {
                        auto top = pq.top();
                        pq.pop();
                        if (top.first <= distance_to_kth_neighbor[q]) {
                            hits++;
                            // only need to record a hit... this actually makes things easier later on
                            nn.push_back(top.second);
                        }
                    }
                    __atomic_fetch_add(&total_hits, hits, __ATOMIC_RELAXED);
                });

                std::cout << "Shard search with ef-search = " << ef_search << " total hits " << total_hits <<
                        " total timer took " << total.total_duration.count() << std::endl;

                ef_search_param_id++;
            }

            std::cout << "Finished searches in bucket " << b << std::endl;
        });
    }

    return shard_searches;
}



std::string ShardSearch::Serialize() const {
    std::stringstream out;
    size_t num_shards = neighbors.size();
    size_t num_queries = neighbors[0].size();
    out << ef_search << " " << num_shards << " " << num_queries << "\n";
    for (size_t b = 0; b < num_shards; ++b) {
        for (size_t q = 0; q < num_queries; ++q) {
            for (const uint32_t neighbor : neighbors[b][q]) {
                out << neighbor << " ";
            }
        }
        out << "\n";
    }
    for (const auto& tq : time_query_in_shard) {
        for (double x : tq) { out << x << " "; }
        out << "\n";
    }
    return out.str();
}

ShardSearch ShardSearch::Deserialize(std::ifstream& in) {
    ShardSearch s;
    int num_shards, num_queries;
    std::string line;
    std::getline(in, line);
    std::istringstream iss(line);
    iss >> s.ef_search >> num_shards >> num_queries;
    s.neighbors.assign(num_shards, std::vector<std::vector<uint32_t>>(num_queries));
    for (int b = 0; b < num_shards; ++b) {
        for (int q = 0; q < num_queries; ++q) {
            std::getline(in, line);
            std::istringstream line_stream(line);
            int neighbor;
            while (line_stream >> neighbor) {
                s.neighbors[b][q].push_back(neighbor);
            }
        }
    }

    for (int b = 0; b < num_shards; ++b) {
        std::getline(in, line);
        std::istringstream line_stream(line);
        s.time_query_in_shard.emplace_back();
        double time;
        while (line_stream >> time) { s.time_query_in_shard.back().push_back(time); }
    }
    return s;
}

void SerializeShardSearches(const std::vector<ShardSearch>& shard_searches, const std::string& output_file) {
    std::ofstream out(output_file);
    out << shard_searches.size() << std::endl;
    for (const ShardSearch& search : shard_searches) {
        out << "S" << std::endl;
        out << search.Serialize();
    }
}

std::vector<ShardSearch> DeserializeShardSearches(const std::string& input_file) {
    std::ifstream in(input_file);
    size_t num_searches;
    std::string header;
    std::getline(in, header);
    std::istringstream iss(header);
    iss >> num_searches;
    std::vector<ShardSearch> shard_searches;
    for (size_t i = 0; i < num_searches; ++i) {
        std::getline(in, header);
        if (header != "S") std::cout << "search config doesn't start with marker S. Instead: " << header << std::endl;
        ShardSearch s = ShardSearch::Deserialize(in);
        shard_searches.push_back(std::move(s));
    }
    return shard_searches;
}
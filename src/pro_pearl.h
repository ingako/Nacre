#ifndef __PRO_PEARL_H__
#define __PRO_PEARL_H__

#include "PEARL/src/cpp/pearl.h"

class pro_pearl : public pearl {

    public:

        pro_pearl(int num_trees,
                  int max_num_candidate_trees,
                  int repo_size,
                  int edit_distance_threshold,
                  int kappa_window_size,
                  int lossy_window_size,
                  int reuse_window_size,
                  int arf_max_features,
                  double bg_kappa_threshold,
                  double cd_kappa_threshold,
                  double reuse_rate_upper_bound,
                  double warning_delta,
                  double drift_delta,
                  double drift_tension);

        virtual int predict();
        virtual void adapt_state(const vector<int>& drifted_tree_pos_list);
        virtual shared_ptr<pearl_tree> make_pearl_tree(int tree_pool_id);
        virtual void init();

        int find_last_actual_drift_point();
        void select_candidate_trees_proactively();
        void adapt_state_proactively();
        bool get_drift_detected();
        void set_expected_drift_prob(double p);
        vector<int> get_drifted_tree_positions();

    private:

        double drift_tension = 0.5;
        bool drift_detected = false;
        int num_max_backtrack_instances = 100000000; // TODO
        int num_instances_seen = 0;
        deque<Instance*> backtrack_instances;
        deque<shared_ptr<pearl_tree>> backtrack_drifted_trees;
        deque<shared_ptr<pearl_tree>> backtrack_swapped_trees;
        deque<long> drifted_points;
        vector<int> drifted_tree_positions;

        static bool compare_kappa_arf(shared_ptr<arf_tree>& tree1,
                                      shared_ptr<arf_tree>& tree2);

};

#endif

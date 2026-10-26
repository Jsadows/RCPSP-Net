#include <torch/extension.h>
#include <vector>
#include <algorithm>
#include <stdexcept>

class FastSSGS {
public:
    FastSSGS(const std::vector<int>& capacities, 
             const std::vector<int>& durations,
             const std::vector<std::vector<int>>& demands,
             const std::vector<std::vector<int>>& successors,
             const std::vector<int>& in_degrees,
             int horizon)
        : capacities_(capacities), 
          durations_(durations), 
          demands_(demands), 
          successors_(successors),
          initial_in_degrees_(in_degrees),
          horizon_(horizon) 
    {
        num_jobs_ = durations_.size();
        num_resources_ = capacities_.size();
        preprocess_all();
        reset();
    }

    void reset() {
        start_times_.assign(num_jobs_, -1);
        finish_times_.assign(num_jobs_, -1);
        resource_usage_.assign(horizon_ * num_resources_, 0);
        es_tracker_.assign(num_jobs_, 0);
        in_degrees_ = initial_in_degrees_;
        
        action_mask_.assign(num_jobs_, false);
        for (int i = 0; i < num_jobs_; ++i) {
            if (in_degrees_[i] == 0) {
                action_mask_[i] = true;
            }
        }
        
        scheduled_count_ = 0;
        cmax_ = 0;
        violations_ = 0;
    }

    torch::Tensor get_mask() {
        auto options = torch::TensorOptions().dtype(torch::kBool);
        std::vector<uint8_t> byte_mask(num_jobs_);
        for(int i = 0; i < num_jobs_; ++i) {
            byte_mask[i] = action_mask_[i] ? 1 : 0;
        }
        return torch::from_blob(byte_mask.data(), {num_jobs_}, options).clone();
    }

    bool step(int j) {
        if (!action_mask_[j]) {
            throw std::runtime_error("Proba wykonania zablokowanego zadania!");
        }

        int es = es_tracker_[j];
        int duration = durations_[j];

        if (duration == 0) {
            start_times_[j] = es;
            finish_times_[j] = es;
        } else {
            int start_t = find_time_window(j, es, duration);
            if (start_t == -1) {
                violations_++;
                start_times_[j] = horizon_;
                finish_times_[j] = horizon_;
                action_mask_[j] = false;
                scheduled_count_++;
                return false;
            }
            start_times_[j] = start_t;
            finish_times_[j] = start_t + duration;
            update_resources(j, start_t, duration);
        }
        if (finish_times_[j] > cmax_) {
            cmax_ = finish_times_[j];
        }
        action_mask_[j] = false;
        scheduled_count_++;

        for (int succ : successors_[j]) {
            if (finish_times_[j] > es_tracker_[succ]) {
                es_tracker_[succ] = finish_times_[j];
            }
            
            in_degrees_[succ]--;
            if (in_degrees_[succ] == 0) {
                action_mask_[succ] = true;
            }
        }
        return true;
    }

    int get_cmax() const { return cmax_; }
    int get_violations() const { return violations_; }
    bool is_done() const { return scheduled_count_ == num_jobs_; }

private:
    int num_jobs_;
    int num_resources_;
    int horizon_;
    int cmax_;
    int violations_;
    int scheduled_count_;
    
    std::vector<int> capacities_;
    std::vector<int> durations_;
    std::vector<std::vector<int>> demands_;
    std::vector<std::vector<int>> successors_;
    std::vector<int> initial_in_degrees_;
    
    std::vector<int> in_degrees_;
    std::vector<int> start_times_;
    std::vector<int> finish_times_;
    std::vector<int> resource_usage_; 
    std::vector<int> es_tracker_; 
    std::vector<bool> action_mask_;
    
    std::vector<std::vector<int>> job_required_resources_;

    void preprocess_all() {
        job_required_resources_.resize(num_jobs_);
        for (int j = 0; j < num_jobs_; ++j) {
            for (int r = 0; r < num_resources_; ++r) {
                if (demands_[j][r] > 0) {
                    job_required_resources_[j].push_back(r);
                }
            }
        }
    }

    int find_time_window(int j, int es, int duration) const {
        int t = es;
        while (t + duration <= horizon_) {
            bool feasible = true;
            int end_window = t + duration - 1;
            for (int tau = end_window; tau >= t; --tau) {
                for (int r : job_required_resources_[j]) { 
                    int usage = resource_usage_[tau * num_resources_ + r];
                    if (usage + demands_[j][r] > capacities_[r]) {
                        feasible = false;
                        t = tau + 1;
                        break; 
                    }
                }
                if (!feasible) break; 
            }
            if (feasible) return t;
        }
        return -1; 
    }
    
    void update_resources(int j, int start_t, int duration) {
        for (int tau = start_t; tau < start_t + duration; ++tau) {
            for (int r : job_required_resources_[j]) {
                resource_usage_[tau * num_resources_ + r] += demands_[j][r];
            }
        }
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<FastSSGS>(m, "FastSSGS")
        .def(pybind11::init<const std::vector<int>&, 
                            const std::vector<int>&, 
                            const std::vector<std::vector<int>>&, 
                            const std::vector<std::vector<int>>&, 
                            const std::vector<int>&, 
                            int>())
        .def("reset", &FastSSGS::reset)
        .def("step", &FastSSGS::step)
        .def("get_mask", &FastSSGS::get_mask)
        .def("get_cmax", &FastSSGS::get_cmax)
        .def("get_violations", &FastSSGS::get_violations)
        .def("is_done", &FastSSGS::is_done);
}
__global__ void updateBestGlobal(float* personal_best,
                                    int* bests_index,
                                    float* global_best, 
                                    int* global_best_index, 
                                    int blocks) {
    *global_best = personal_best[0];
    *global_best_index = 0;
    for (int i = 1; i < blocks; i++) {
        if (personal_best[i] < *global_best) {
            *global_best = personal_best[i];
            *global_best_index = bests_index[i];
        }
    }
}
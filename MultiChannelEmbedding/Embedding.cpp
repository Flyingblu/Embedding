#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "LatentModel.hpp"
#include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	Dataset dataset("latest-lexemes", "E:\\Data\\Code\\Project\\XiaoEmbedding\\data\\latest-lexemes\\", "training_.data", "", "testing_.data", false);
	TaskType task = General;
	MFactorE model(dataset, task, "E:\\Data\\Code\\Project\\XiaoEmbedding\\log\\training", 5, 0.01, 0.1, 0.01, 10);
	model.run(2000);
	model.test_link_prediction();
	return 0;
}
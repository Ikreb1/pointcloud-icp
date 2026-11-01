// Correctness checks, accuracy scenarios and benchmarks for the ICP pipeline.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numbers>
#include <random>

#include <omp.h>

#include "alignment.h"
#include "cloud.h"
#include "icp.h"
#include "kdtree.h"
#include "nearest.h"
#include "timer.h"

constexpr int TEST_SIZE = 10000;

void runScenario(const char* name, const Cloud& source, Search search, float angleDegrees, float noiseSigma, std::mt19937& rng)
{
	const float angle = angleDegrees * std::numbers::pi_v<float> / 180.0f;
	const Eigen::Matrix3f R = rotationZ(angle);
	const Eigen::Vector3f t(0.1f, -0.05f, 0.2f);

	Cloud target = transformCloud(source, R, t);
	if (noiseSigma > 0)
		target = addNoise(target, noiseSigma, rng);

	const IcpResult icp = runIcp(source, target, search, 500, 1e-6f, false);
	const double rotErr = rotationErrorDegrees(icp.transform.R, R);
	const float transErr = (icp.transform.t - t).norm();

	std::cout << name
	          << "  iters " << icp.iterations
	          << "  rmse " << icp.rmse
	          << "  rot err " << rotErr << " deg"
	          << "  trans err " << transErr
	          << "  " << icp.msPerIteration << " ms/iter";
	if (search != Search::BruteForce)
		std::cout << " (+" << icp.buildMs << " ms build)";
	std::cout << (rotErr < 0.5 ? "  OK" : "  FAILED") << "\n";

	const double pct = 100.0 / icp.msPerIteration;
	std::cout << "    per iter: match " << icp.matchMs << " ms (" << icp.matchMs * pct << "%)"
	          << "  align " << icp.alignMs << " ms (" << icp.alignMs * pct << "%)"
	          << "  transform " << icp.transformMs << " ms (" << icp.transformMs * pct << "%)\n";
}

// The tree must find the same nearest distances as brute force. Allows a
// rounding-sized difference: the SIMD path computes dx*dx + dy*dy + dz*dz with
// FMA in a fixed order, which can differ from the scalar code in the last bit.
// Indices can differ only on exact ties, so compare distances.
bool kdTreeMatchesBruteForce(const Cloud& cloud, Search search, int queries, std::mt19937& rng)
{
	const KdTree tree(cloud);
	std::uniform_real_distribution<float> dist(-1.2f, 1.2f); // some queries outside the cloud
	for (int q = 0; q < queries; ++q)
	{
		const Point query{dist(rng), dist(rng), dist(rng)};
		const NearestResult expected = nearestNeighbour(cloud, query);
		const NearestResult actual = search == Search::KdTreeSimd ? tree.nearest(query) : tree.nearestScalar(query);
		if (std::abs(actual.distanceSquared - expected.distanceSquared) > 1e-6f * expected.distanceSquared + 1e-12f)
		{
			std::cout << "MISMATCH at query " << q << ": tree " << actual.distanceSquared << " vs brute " << expected.distanceSquared << "\n";
			return false;
		}
	}
	return true;
}

// Times the nearest-neighbour queries alone (98% of an ICP iteration) for
// different leaf sizes. Queries: the target rotated 10 degrees plus noise,
// like an early ICP iteration. Best of 5 runs, to filter out noise.
void benchmarkLeafSizes(const Cloud& target, std::mt19937& rng)
{
	const Cloud queries = addNoise(transformCloud(target, rotationZ(10.0f * std::numbers::pi_v<float> / 180.0f), {0.1f, -0.05f, 0.2f}), 0.005f, rng);
	const int n = static_cast<int>(queries.points.size());

	std::cout << "\nLeaf size sweep, " << target.points.size() << " points (ms for all queries, best of 5):\n";
	std::cout << "  leaf   build ms   scalar ms   simd ms   simd speedup\n";
	for (int leafSize : {8, 16, 32, 64, 128})
	{
		const auto buildStart = std::chrono::steady_clock::now();
		const KdTree tree(target, leafSize);
		const double buildMs = msSince(buildStart);

		double best[2] = {1e30, 1e30};
		for (int simd = 0; simd < 2; ++simd)
		{
			for (int run = 0; run < 5; ++run)
			{
				const auto start = std::chrono::steady_clock::now();
				double checksum = 0; // use the results so the compiler can't skip the work
#pragma omp parallel for reduction(+ : checksum) schedule(static)
				for (int i = 0; i < n; ++i)
					checksum += simd ? tree.nearest(queries.points[i]).distanceSquared : tree.nearestScalar(queries.points[i]).distanceSquared;
				best[simd] = std::min(best[simd], msSince(start));
				if (checksum < 0)
					std::cout << "";
			}
		}
		std::printf("  %4d   %8.2f   %9.3f   %7.3f   %.2fx\n", leafSize, buildMs, best[0], best[1], best[0] / best[1]);
	}
}

int main()
{
	std::mt19937 rng(42);
	const Cloud small = makeRandomCloud(TEST_SIZE, rng);

	std::cout << omp_get_max_threads() << " threads, leaf size " << KD_LEAF_SIZE << "\n\n";

	std::cout << "Correctness vs brute force, 10000 random queries:\n";
	std::cout << "  k-d tree scalar: " << (kdTreeMatchesBruteForce(small, Search::KdTreeScalar, 10000, rng) ? "same" : "DIFFERENT") << "\n";
	std::cout << "  k-d tree SIMD:   " << (kdTreeMatchesBruteForce(small, Search::KdTreeSimd, 10000, rng) ? "same" : "DIFFERENT") << "\n\n";

	std::cout << "Accuracy, " << TEST_SIZE << " points:\n";
	runScenario("brute  10 deg, clean", small, Search::BruteForce, 10, 0.0f, rng);
	runScenario("simd   10 deg, clean", small, Search::KdTreeSimd, 10, 0.0f, rng);
	runScenario("simd   10 deg, noise", small, Search::KdTreeSimd, 10, 0.005f, rng);
	runScenario("simd   30 deg, clean", small, Search::KdTreeSimd, 30, 0.0f, rng);
	runScenario("simd   45 deg, clean", small, Search::KdTreeSimd, 45, 0.0f, rng);

	for (int size : {100000, 1000000})
	{
		const Cloud big = makeRandomCloud(size, rng);
		std::cout << "\nSpeed, " << size << " points, 10 deg + noise:\n";
		runScenario("scalar", big, Search::KdTreeScalar, 10, 0.005f, rng);
		runScenario("simd  ", big, Search::KdTreeSimd, 10, 0.005f, rng);
		benchmarkLeafSizes(big, rng);
	}
}
